import api.{type AgentResponse, send_message}
import gleam/string
import meditation

import cartesia
import dom
import gleam/dynamic/decode
import gleam/list
import gleam/option.{type Option, None, Some}
import lustre
import lustre/attribute
import lustre/effect.{type Effect}
import lustre/element.{type Element}
import lustre/element/html
import lustre/event
import mork
import mork/to_lustre
import theme.{type Theme, Dark, Light, System}

// --- TYPES ---

pub type Message {
  Message(role: String, content: String)
}

pub type Model {
  Model(
    chat_history: List(Message),
    is_streaming: Bool,
    input_text: String,
    loading: Bool,
    answer: Option(String),
    transcript: Option(String),
    thread_id: Option(String),
    theme: Theme,
    show_meditation: Bool,
    tts: cartesia.Model,
  )
}

pub type Msg {
  Noop
  UserTyped(String)
  UserRequestedAudio
  AudioStarted
  AudioEnded
  ReceiveChatResponse(AgentResponse)
  SendMessage
  SetTheme(Theme)
  HideMeditationScreen
  CartesiaMsg(cartesia.Msg)
}

// --- FFI (Foreign Function Interface) ---

@external(javascript, "./ffi/auto_height_ffi.mjs", "auto_height")
fn auto_height(id: String) -> Nil

@external(javascript, "./ffi/auto_height_ffi.mjs", "reset_height")
fn reset_height(id: String) -> Nil

// --- APP LOGIC ---

fn init(_flags) -> #(Model, Effect(Msg)) {
  let cartesia_init =
    cartesia.Model(
      ws: None,
      is_connected: False,
      input_text: "",
      status_message: "Disconnected",
      pending_chunks: [],
      current_context_index: 0,
    )
  let #(tts, eff) = cartesia.update(cartesia_init, cartesia.Connect)
  #(
    Model(
      chat_history: [],
      is_streaming: False,
      input_text: "",
      loading: False,
      thread_id: None,
      answer: None,
      transcript: None,
      theme: System,
      show_meditation: False,
      tts: tts,
    ),
    effect.map(eff, CartesiaMsg),
  )
}

fn update(model: Model, msg: Msg) -> #(Model, Effect(Msg)) {
  case msg {
    Noop -> #(model, effect.none())

    CartesiaMsg(submsg) -> {
      let #(tts, eff) = cartesia.update(model.tts, submsg)
      #(Model(..model, tts: tts), effect.map(eff, CartesiaMsg))
    }

    UserTyped(val) -> #(
      Model(..model, input_text: val),
      effect.from(fn(_) { auto_height("user-input") }),
    )

    UserRequestedAudio -> {
      let #(tts, eff) = cartesia.update(model.tts, cartesia.GenerateAudio)
      echo tts
      #(
        Model(..model, is_streaming: True, tts: tts),
        effect.map(eff, CartesiaMsg),
      )
    }

    AudioStarted -> #(Model(..model, is_streaming: True), effect.none())

    AudioEnded -> #(Model(..model, is_streaming: False), effect.none())

    ReceiveChatResponse(msg) -> {
      #(
        Model(
          ..model,
          chat_history: list.append(model.chat_history, [
            Message(role: "assistant", content: msg.reply),
          ]),
          loading: False,
          thread_id: Some(msg.thread_id),
          answer: msg.answer,
          transcript: msg.transcript,
          show_meditation: option.is_some(msg.answer)
            && option.is_some(msg.transcript),
          tts: cartesia.Model(
            ..model.tts,
            input_text: option.unwrap(msg.transcript, ""),
          ),
        ),
        // dom.scroll_to_bottom_delayed("chat-ancor"),
        effect.none(),
      )
    }

    SendMessage ->
      case model.loading, string.trim(model.input_text) {
        True, _ -> #(model, effect.none())
        _, "" -> #(model, effect.none())
        False, user_message -> {
          #(
            Model(
              ..model,
              chat_history: list.append(model.chat_history, [
                Message(role: "user", content: user_message),
              ]),
              is_streaming: False,
              input_text: "",
              loading: True,
            ),
            effect.batch([
              // dom.scroll_to_bottom_delayed("chat-ancor"),
              effect.from(fn(_) { reset_height("user-input") }),
              effect.map(send_message(user_message, model.thread_id), fn(res) {
                let assert Ok(ar) = res
                ReceiveChatResponse(ar)
              }),
            ]),
          )
        }
      }

    SetTheme(new_theme) -> {
      let theme_class = case new_theme {
        Dark -> "dark"
        Light -> "light"
        System -> ""
      }
      #(Model(..model, theme: new_theme), dom.sync_body_class(theme_class))
    }

    HideMeditationScreen -> #(
      Model(..model, show_meditation: False),
      effect.none(),
    )
  }
}

// --- VIEW ---

fn view_loading_indicator() -> Element(Msg) {
  html.div([attribute.class("flex justify-start gap-3 items-end py-2")], [
    // Avatar Placeholder: Matches the reactive message avatar
    html.div(
      [
        attribute.class(
          "w-8 h-8 rounded-full flex-shrink-0 transition-colors duration-500 "
          <> "bg-neo-mint/20 dark:bg-neo-mint/10 border border-neo-mint/30 dark:border-neo-mint/20",
        ),
      ],
      [],
    ),

    // Bubble: Uses the same semantic mapping as the agent message
    html.div(
      [
        attribute.class(
          "flex items-center space-x-2 px-5 py-4 rounded-2xl rounded-bl-none shadow-sm border "
          <> "bg-bubble-agent-bg border-bubble-agent-border",
        ),
      ],
      [
        // Pulsing Dots: Using the agent text color with opacity for consistency
        dot("animate-bounce"),
        dot("animate-bounce [animation-delay:0.2s]"),
        dot("animate-bounce [animation-delay:0.4s]"),
      ],
    ),
  ])
}

// Helper to keep the dots consistent and theme-aware
fn dot(extra_class: String) -> Element(Msg) {
  html.div(
    [
      attribute.class(
        "w-1.5 h-1.5 rounded-full transition-colors duration-500 "
        <> "bg-bubble-agent-text/40 "
        <> extra_class,
      ),
    ],
    [],
  )
}

fn view_message(m: Message) -> Element(Msg) {
  let is_user = m.role == "user"

  let wrapper_class = case is_user {
    True -> "flex justify-end w-full"
    False -> "flex justify-start gap-3 items-end"
  }

  let bubble_class = case is_user {
    True ->
      "max-w-[85%] px-5 py-3 text-sm leading-relaxed whitespace-pre-wrap rounded-2xl rounded-br-none shadow-md "
      <> "bg-bubble-user-bg text-bubble-user-text"

    False ->
      "max-w-[85%] px-5 py-3 text-sm leading-relaxed whitespace-pre-wrap rounded-2xl rounded-bl-none shadow-sm border "
      <> "bg-bubble-agent-bg text-bubble-agent-text border-bubble-agent-border"
  }

  html.div([attribute.class(wrapper_class)], [
    // Agent Avatar Icon: Reactive background and border
    case is_user {
      False ->
        html.div(
          [
            attribute.class(
              "w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-lg transition-colors duration-500 "
              <> "bg-neo-mint/20 dark:bg-neo-mint/10 border border-neo-mint/30 dark:border-neo-mint/20",
            ),
          ],
          [element.text("🌿")],
        )
      True -> html.text("")
    },

    html.div(
      [attribute.class(bubble_class)],
      m.content
        |> mork.parse
        |> to_lustre.to_lustre,
    ),
  ])
}

fn view(model: Model) -> Element(Msg) {
  case model.show_meditation {
    True -> {
      let assert Some(quote) = model.answer
      meditation.view_meditation_screen(
        quote,
        UserRequestedAudio,
        HideMeditationScreen,
      )
    }
    False ->
      html.div(
        [
          attribute.class(
            "min-h-screen bg-bg-main font-body text-text-base transition-colors duration-500 "
            <> "lg:flex lg:items-center lg:justify-center lg:p-4",
            // Centering only on large screens
          ),
        ],
        [
          html.div(
            [
              attribute.class(
                // MOBILE: Full width/height, no corners, no border
                "flex flex-col w-full h-svh bg-bg-header/90 backdrop-blur-md shadow-2xl "
                // DESKTOP (lg): Fixed size, rounded corners, border
                <> "lg:h-[850px] lg:max-w-md lg:rounded-[3rem] lg:border lg:border-deep-moss/10 lg:relative",
              ),
            ],
            [
              // FIXED HEADER: Pinned to top
              html.header(
                [
                  attribute.class(
                    "flex flex-col items-center justify-center pt-8 pb-4 px-6 text-center border-b border-deep-moss/5 bg-bg-header/50 lg:rounded-[3rem]",
                  ),
                ],
                [
                  html.div(
                    [
                      attribute.class(
                        "text-warm-sand text-8xl leading-none h-[60px] flex col items-center justify-center",
                      ),
                    ],
                    [
                      html.i(
                        [attribute.class("icon-pulse-lotus inline-flex")],
                        [],
                      ),
                    ],
                  ),
                  html.p(
                    [
                      attribute.class(
                        "text-[10px] font-semibold tracking-[0.2em] text-gold-leaf mb-1",
                      ),
                    ],
                    [
                      element.text("PULSE LOTUS"),
                    ],
                  ),
                  html.h1(
                    [
                      attribute.class(
                        "text-xl font-header font-extralight text-deep-moss tracking-tight",
                      ),
                    ],
                    [element.text("TAILOR YOUR SESSION")],
                  ),
                ],
              ),

              // SCROLLABLE MESSAGES: Fills all remaining space
              html.div(
                [
                  attribute.class(
                    "flex-1 overflow-hidden bg-tactile-nature shadow-inner transition-colors duration-500",
                    // Fills gap between header and footer
                  ),
                ],
                [
                  case model.chat_history {
                    [] ->
                      html.p(
                        [
                          attribute.class(
                            "text-welcome-text text-sm text-center italic mt-20",
                          ),
                        ],
                        [
                          element.text("How are you feeling in this moment?"),
                        ],
                      )
                    _ -> html.text("")
                  },

                  html.div(
                    [
                      attribute.class(
                        "flex flex-col-reverse gap-y-6 p-6 overflow-y-auto max-h-full scrollbar-hide scroll-smooth",
                      ),
                    ],
                    model.chat_history
                      |> list.reverse
                      |> list.map(view_message)
                      |> list.prepend(case model.loading {
                        True -> view_loading_indicator()
                        False -> html.text("")
                      }),
                  ),
                ],
              ),

              // FIXED FOOTER (INPUT): Pinned to bottom
              html.footer(
                [
                  attribute.class(
                    "p-4 pb-8 lg:pb-6 bg-bg-header/80 backdrop-blur-lg border-t border-deep-moss/5 lg:rounded-[3rem]",
                  ),
                ],
                [
                  html.div(
                    [
                      attribute.class(
                        "flex gap-2 bg-bg-main border border-gold-leaf/20 p-1.5 rounded-[1rem] shadow-inner focus-within:border-gold-leaf/50 transition-colors",
                      ),
                    ],
                    [
                      html.textarea(
                        [
                          attribute.id("user-input"),
                          attribute.placeholder(
                            "Tell me more about your feelings...",
                          ),
                          attribute.class(
                            "flex-1 bg-transparent px-5 py-3 text-sm outline-none resize-none",
                          ),
                          attribute.rows(1),
                          event.on_input(UserTyped),
                          event.advanced("keydown", {
                            use key <- decode.field("key", decode.string)
                            use shift <- decode.field("shiftKey", decode.bool)
                            let handler =
                              event.handler(
                                dispatch: SendMessage,
                                prevent_default: True,
                                stop_propagation: False,
                              )
                            case shift, key {
                              False, "Enter" -> decode.success(handler)
                              _, _ -> decode.failure(handler, "SendMessage")
                            }
                          }),
                        ],
                        model.input_text,
                      ),
                      html.button(
                        [
                          event.on_click(SendMessage),
                          attribute.disabled(model.loading),
                          attribute.class(
                            "icon-paper-plane rounded-[.7rem] text-gold-leaf cursor-pointer text-4xl hover:shadow-xl hover:text-off-white transition-all bg-bubble-user-bg text-bubble-user-text",
                          ),
                        ],
                        [],
                      ),
                    ],
                  ),
                  // view_theme_toggle(model.theme, fn(theme) { SetTheme(theme) }),
                ],
              ),
            ],
          ),
        ],
      )
  }
}

pub fn main() {
  let app = lustre.application(init, update, view)
  let assert Ok(_) = lustre.start(app, "#app", Nil)
}
