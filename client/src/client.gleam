import api.{type AgentResponse, send_message}
import gleam/list
import gleam/option.{type Option, None, Some}
import lustre
import lustre/attribute
import lustre/effect.{type Effect}
import lustre/element.{type Element}
import lustre/element/html
import lustre/event

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
  )
}

pub type Msg {
  UserTyped(String)
  UserRequestedAudio
  AudioStarted
  AudioEnded
  ReceiveChatResponse(AgentResponse)
  SendMessage
  // New message type
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
    ),
    effect.none(),
  )
}

fn update(model: Model, msg: Msg) -> #(Model, Effect(Msg)) {
  case msg {
    UserTyped(val) -> #(Model(..model, input_text: val), effect.none())

    UserRequestedAudio -> #(
      Model(..model, is_streaming: True),
      effect.from(fn(_) { init_stream_ffi() }),
    )

    AudioStarted -> #(Model(..model, is_streaming: True), effect.none())

    AudioEnded -> #(Model(..model, is_streaming: False), effect.none())

    ReceiveChatResponse(msg) -> #(
      Model(
        chat_history: list.append(model.chat_history, [
          Message(role: "assistant", content: msg.message),
        ]),
        is_streaming: model.is_streaming,
        input_text: model.input_text,
        loading: False,
        session_id: Some(msg.session_id),
        transcript: msg.transcript,
      ),
      effect.none(),
    )

    SendMessage -> #(
      Model(
        ..model,
        chat_history: [
          Message(role: "user", content: model.input_text),
          ..model.chat_history
        ],
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
}

// --- VIEW ---

fn view(model: Model) -> Element(Msg) {
  html.div(
    [attribute.class("min-h-screen flex items-center justify-center px-4")],
    [
      html.div(
        [
          attribute.class(
            "w-full max-w-3xl bg-mind-surface/80 backdrop-blur-md rounded-2xl shadow-xl border border-slate-700/60 p-6 flex flex-col gap-4",
          ),
        ],
        [
          html.header(
            [attribute.class("flex items-center justify-between mb-2")],
            [
              html.div([], [
                html.h1(
                  [attribute.class("text-xl font-semibold text-sky-300")],
                  [element.text("Mindfulness AI")],
                ),
                html.p([attribute.class("text-sm text-slate-300")], [
                  element.text(
                    "Share how you feel, and receive a gentle, guided response.",
                  ),
                ]),
              ]),
            ],
          ),

          html.div(
            [
              attribute.class(
                "flex-1 min-h-[320px] max-h-[420px] overflow-y-auto space-y-3 pr-1",
              ),
            ],
            [
              case model.chat_history {
                [] ->
                  html.p([attribute.class("text-slate-400 text-sm")], [
                    element.text(
                      "Start by telling the guide what you are going through, for example: “I feel anxious about work and cannot relax.”",
                    ),
                  ])
                _ -> html.text("")
              },

              html.div(
                [],
                model.chat_history
                  |> list.map(fn(m) {
                    let justify_class = case m.role {
                      "user" -> "justify-end"
                      _ -> "justify-start"
                    }
                    let bg_class = case m.role {
                      "user" -> "bg-mind-accent/90 text-slate-950"
                      _ ->
                        "bg-slate-800/80 text-slate-100 border border-slate-700/70"
                    }
                    html.div([attribute.class("flex " <> justify_class)], [
                      html.div(
                        [
                          attribute.class(
                            "max-w-[80%] rounded-2xl px-3 py-2 text-sm leading-relaxed whitespace-pre-wrap "
                            <> bg_class,
                          ),
                        ],
                        [element.text(m.content)],
                      ),
                    ])
                  }),
              ),

              case model.loading {
                True ->
                  html.div(
                    [attribute.class("flex justify-center items-center py-2")],
                    [
                      html.div(
                        [attribute.class("flex items-center space-x-2")],
                        [
                          html.div(
                            [
                              attribute.class(
                                "w-2 h-2 bg-sky-400 rounded-full animate-pulse",
                              ),
                            ],
                            [],
                          ),
                          html.div(
                            [
                              attribute.class(
                                "w-2 h-2 bg-sky-400 rounded-full animate-pulse delay-100",
                              ),
                            ],
                            [],
                          ),
                          html.div(
                            [
                              attribute.class(
                                "w-2 h-2 bg-sky-400 rounded-full animate-pulse delay-200",
                              ),
                            ],
                            [],
                          ),
                        ],
                      ),
                    ],
                  )
                False -> html.text("")
              },
            ],
          ),

          html.div([attribute.class("mt-2 flex gap-2")], [
            html.input([
              attribute.type_("text"),
              event.on_change(UserTyped),
              // event.on_keydown(fn(_) { SendMessage }),
              attribute.value(model.input_text),
              attribute.placeholder("How are you feeling?"),
              attribute.class(
                "flex-1 rounded-xl bg-slate-900/60 border border-slate-700/70 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-mind-accent focus:border-transparent",
              ),
            ]),

            html.button(
              [
                event.on_click(SendMessage),
                attribute.disabled(model.loading),
                attribute.class(
                  "px-4 py-2 rounded-xl bg-mind-accent text-slate-950 text-sm font-medium hover:bg-sky-400 disabled:opacity-60 disabled:cursor-not-allowed transition-colors",
                ),
              ],
              [
                element.text(case model.loading {
                  True -> "Sending..."
                  False -> "Send"
                }),
              ],
            ),
            // ,

          // case transcript {
          //   Some(_) -> html.div([attribute.class("flex justify-center mt-2")], [
          //     audio([
          //       ref(audio_ref),
          //       attribute.class("hidden"),
          //       style("display", "none"),
          //       on_can_play(fn (_) {io.println("Audio can play")}),
          //       on_ended(fn (_) {io.println("Audio ended")})
          //     ], [])
          //   ])
          //   None -> html_nil()
          // }
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
