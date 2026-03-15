import api.{type AgentResponse, send_message}
import gleam/dynamic/decode
import gleam/list
import gleam/option.{type Option, None, Some}
import lustre
import lustre/attribute
import lustre/effect.{type Effect}
import lustre/element.{type Element}
import lustre/element/html
import lustre/event
import theme.{type Theme, Dark, Light, System, view_theme_toggle}

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
    transcript: Option(String),
    session_id: Option(String),
    theme: Theme,
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
}

// --- FFI (Foreign Function Interface) ---

// This maps to a function in your audio_bridge.js
@external(javascript, "./audio_ffi.mjs", "init_cartesia_stream")
fn init_stream_ffi() -> Nil

// --- APP LOGIC ---

fn init(_flags) -> #(Model, Effect(Msg)) {
  #(
    Model(
      chat_history: [],
      is_streaming: False,
      input_text: "",
      loading: False,
      session_id: None,
      transcript: None,
      theme: System,
    ),
    effect.none(),
  )
}

fn update(model: Model, msg: Msg) -> #(Model, Effect(Msg)) {
  case echo msg {
    Noop -> #(model, effect.none())

    UserTyped(val) -> #(Model(..model, input_text: val), effect.none())

    UserRequestedAudio -> #(
      Model(..model, is_streaming: True),
      effect.from(fn(_) { init_stream_ffi() }),
    )

    AudioStarted -> #(Model(..model, is_streaming: True), effect.none())

    AudioEnded -> #(Model(..model, is_streaming: False), effect.none())

    ReceiveChatResponse(msg) -> #(
      Model(
        ..model,
        chat_history: list.append(model.chat_history, [
          Message(role: "assistant", content: msg.message),
        ]),
        loading: False,
        session_id: Some(msg.session_id),
        transcript: msg.transcript,
      ),
      effect.none(),
    )

    SendMessage ->
      case model.loading {
        True -> #(model, effect.none())
        False -> #(
          Model(
            ..model,
            chat_history: list.append(model.chat_history, [
              Message(role: "user", content: model.input_text),
            ]),
            is_streaming: False,
            input_text: "",
            loading: True,
          ),
          effect.map(send_message(model.input_text, model.session_id), fn(res) {
            let assert Ok(ar) = res
            ReceiveChatResponse(ar)
          }),
        )
      }

    SetTheme(new_theme) -> #(Model(..model, theme: new_theme), effect.none())
  }
}

// --- VIEW ---

fn view(model: Model) -> Element(Msg) {
  // Determine if the dark class should be applied
  let theme_class = case model.theme {
    Dark -> "dark "
    Light -> ""
    System -> ""
  }

  html.div(
    // Applying the main background and base text colors from our Tailwind theme
    [
      attribute.class(
        theme_class
        <> "min-h-screen flex items-center justify-center p-4 bg-bg-main font-body text-text-base",
      ),
    ],
    [
      html.div(
        [
          attribute.class(
            "w-full max-w-md bg-bg-header/90 backdrop-blur-md rounded-[2.5rem] shadow-2xl border border-deep-moss/10 p-6 flex flex-col gap-4 relative",
          ),
        ],
        [
          // view_theme_toggle(model.theme, fn(theme) { SetTheme(theme) }),
          // Header Section
          html.header(
            [
              attribute.class(
                "flex flex-col items-center justify-center mb-4 mt-2 text-center",
              ),
            ],
            [
              html.div([attribute.class("pulse-lotus gold-linear")], []),
              html.p(
                [
                  attribute.class(
                    "text-xs font-semibold tracking-widest text-gold-leaf mb-2",
                  ),
                ],
                [
                  // element.text("ETHEREAL FLORA"),
                  element.text("PULSE LOTUS"),
                ],
              ),
              html.h1(
                [
                  attribute.class(
                    "text-2xl font-header font-extralight text-deep-moss tracking-tight",
                  ),
                ],
                [element.text("TAILOR YOUR SESSION")],
              ),
            ],
          ),

          // Chat History Area
          html.div(
            [
              attribute.class(
                "flex-1 min-h-[400px] max-h-[500px] overflow-y-auto space-y-4 pr-2 scrollbar-thin",
              ),
            ],
            [
              case model.chat_history {
                [] ->
                  html.p(
                    [
                      attribute.class(
                        "text-charcoal/50 text-sm text-center italic mt-10",
                      ),
                    ],
                    [
                      element.text(
                        "Welcome. Start by sharing a bit about how you're feeling today.",
                      ),
                    ],
                  )
                _ -> html.text("")
              },

              html.div(
                [attribute.class("space-y-4")],
                model.chat_history
                  |> list.map(fn(m) {
                    let is_user = case m.role {
                      "user" -> True
                      _ -> False
                    }

                    let wrapper_class = case is_user {
                      True -> "flex justify-end"
                      False -> "flex justify-start gap-2"
                    }

                    let bubble_class = case is_user {
                      True ->
                        "max-w-[85%] px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap bg-neo-mint text-deep-moss rounded-2xl rounded-br-sm shadow-sm"
                      False ->
                        "max-w-[85%] px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap bg-bg-main border border-deep-moss/10 text-charcoal rounded-2xl rounded-bl-sm shadow-sm"
                    }

                    html.div([attribute.class(wrapper_class)], [
                      // Agent Avatar Icon
                      case is_user {
                        False ->
                          html.div(
                            [
                              attribute.class(
                                "w-8 h-8 rounded-full bg-neo-mint/20 flex-shrink-0 flex items-center justify-center text-deep-moss border border-neo-mint/50",
                              ),
                            ],
                            [element.text("🌿")],
                          )
                        True -> html.text("")
                      },
                      html.div([attribute.class(bubble_class)], [
                        element.text(m.content),
                      ]),
                    ])
                  }),
              ),

              // Loading Animation (Pulsing Dots)
              case model.loading {
                True ->
                  html.div([attribute.class("flex justify-start gap-2 py-2")], [
                    html.div(
                      [
                        attribute.class(
                          "w-8 h-8 rounded-full bg-neo-mint/20 flex-shrink-0",
                        ),
                      ],
                      [],
                    ),
                    html.div(
                      [
                        attribute.class(
                          "flex items-center space-x-1.5 bg-bg-main border border-deep-moss/10 px-4 py-3 rounded-2xl rounded-bl-sm",
                        ),
                      ],
                      [
                        html.div(
                          [
                            attribute.class(
                              "w-1.5 h-1.5 bg-deep-moss/40 rounded-full animate-bounce",
                            ),
                          ],
                          [],
                        ),
                        html.div(
                          [
                            attribute.class(
                              "w-1.5 h-1.5 bg-deep-moss/40 rounded-full animate-bounce delay-100",
                            ),
                          ],
                          [],
                        ),
                        html.div(
                          [
                            attribute.class(
                              "w-1.5 h-1.5 bg-deep-moss/40 rounded-full animate-bounce delay-200",
                            ),
                          ],
                          [],
                        ),
                      ],
                    ),
                  ])
                False -> html.text("")
              },
            ],
          ),

          // Input Area (Pill-shaped as per mock)
          html.div([attribute.class("mt-2")], [
            html.div(
              [
                attribute.class(
                  "flex gap-2 bg-bg-header backdrop-blur-md border border-gold-leaf/30 p-1.5 rounded-full shadow-inner",
                ),
              ],
              [
                html.input([
                  attribute.type_("text"),
                  event.on_input(UserTyped),
                  event.advanced("keydown", {
                    use key <- decode.field("key", decode.string)
                    let handler =
                      event.handler(
                        dispatch: SendMessage,
                        prevent_default: True,
                        stop_propagation: False,
                      )
                    case key {
                      "Enter" -> decode.success(handler)
                      _ -> decode.failure(handler, "SendMessage")
                    }
                  }),
                  attribute.value(model.input_text),
                  attribute.placeholder("Tell me more about your feelings..."),
                  attribute.class(
                    "flex-1 bg-transparent px-4 py-2 text-sm text-text-base placeholder:text-charcoal/40 focus:outline-none",
                  ),
                ]),

                html.button(
                  [
                    event.on_click(SendMessage),
                    attribute.disabled(model.loading),
                    attribute.class(
                      "px-6 py-2 rounded-full bg-neo-mint text-deep-moss text-sm font-medium hover:bg-deep-moss hover:text-off-white disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300",
                    ),
                  ],
                  [
                    element.text(case model.loading {
                      True -> "..."
                      False -> "send"
                    }),
                  ],
                ),
              ],
            ),
          ]),
        ],
      ),
    ],
  )
}

pub fn main() {
  let app = lustre.application(init, update, view)
  let assert Ok(_) = lustre.start(app, "#app", Nil)
}
