import gleam/dynamic/decode.{type Dynamic}
import gleam/json
import gleam/option.{type Option, None, Some}

// import lustre
// import lustre/attribute
import lustre/effect.{type Effect}

// import lustre/element.{type Element}
// import lustre/element/html
// import lustre/event
import lustre_websocket as ws

// --- MODEL ---

pub type Model {
  Model(
    ws: Option(ws.WebSocket),
    is_connected: Bool,
    input_text: String,
    status_message: String,
  )
}

// pub fn init(_) -> #(Model, Effect(Msg)) {
//   let api_key = get_cartesia_key()
//   echo api_key

//   let initial_model =
//     Model(
//       ws: None,
//       is_connected: False,
//       input_text: "Breathe in, breathe out...",
//       status_message: "Disconnected",
//     )

//   // Cartesia requires auth and version directly in the WebSocket URL 
//   let cartesia_url =
//     "wss://api.cartesia.ai:443/tts/websocket?api_key="
//     <> api_key
//     <> "&cartesia_version=2024-06-10"

//   #(initial_model, ws.init(cartesia_url, WsEvent))
// }

// --- MESSAGES ---

pub type Msg {
  WsEvent(ws.WebSocketEvent)
  UpdateInput(String)
  GenerateAudio
  Connect
}

pub type CartesiaMessage {
  AudioChunk(data: String)
  StreamDone(request_id: String)
}

pub fn message_decoder() -> decode.Decoder(CartesiaMessage) {
  decode.one_of(
    // Decoder 1: Look for an audio chunk
    decode.at(["type"], decode.string)
      |> decode.then(fn(msg_type) {
        case msg_type {
          "chunk" -> {
            use data <- decode.field("data", decode.string)
            decode.success(AudioChunk(data:))
          }
          "done" -> {
            use id <- decode.field("request_id", decode.string)
            decode.success(StreamDone(id))
          }
          _ -> decode.failure(AudioChunk(data: ""), "Unknown message type")
        }
      }),
    [],
  )
}

// Function to build the sophisticated voice payload
fn build_voice_payload(
  voice_id: String,
  speed: Float,
  emotion: String,
) -> json.Json {
  json.object([
    #("mode", json.string("id")),
    #("id", json.string(voice_id)),
    #(
      "settings",
      json.object([
        #("speed", json.float(speed)),
        #("emotion", json.array([emotion], of: json.string)),
      ]),
    ),
  ])
}

// --- FFI (Foreign Function Interface) ---

@external(javascript, "./ffi/env_ffi.mjs", "get_cartesia_key")
fn get_cartesia_key() -> String

@external(javascript, "./ffi/audio_ffi.mjs", "init_audio")
fn init_audio() -> Nil

@external(javascript, "./ffi/audio_ffi.mjs", "play_chunk")
fn play_chunk(base64: String) -> Nil

@external(javascript, "./ffi/audio_ffi.mjs", "get_analyser")
fn get_analyser() -> Dynamic

@external(javascript, "./ffi/visualizer_ffi.mjs", "mount_visualizer")
fn mount_visualizer(id: String, analyser: Dynamic) -> Nil

// --- UPDATE ---

pub fn update(model: Model, msg: Msg) -> #(Model, Effect(Msg)) {
  case echo msg {
    Connect -> {
      let api_key = get_cartesia_key()
      let cartesia_url =
        "wss://api.cartesia.ai:443/tts/websocket?api_key="
        <> api_key
        <> "&cartesia_version=2024-06-10"
      #(model, ws.init(cartesia_url, WsEvent))
    }
    WsEvent(ws.InvalidUrl) -> {
      #(
        Model(..model, status_message: "Error: Invalid Cartesia URL"),
        effect.none(),
      )
    }

    WsEvent(ws.OnOpen(socket)) -> {
      #(
        Model(
          ..model,
          ws: Some(socket),
          is_connected: True,
          status_message: "Connected to Voice Model",
        ),
        effect.from(fn(_) { init_audio() }),
      )
    }

    WsEvent(ws.OnTextMessage(json_string)) -> {
      // Cartesia streams audio chunks as JSON payloads containing base64 audio.
      // NOTE: Dispatch to a JS FFI port here to queue chunks in a Web Audio API buffer.
      let result = json.parse(json_string, message_decoder())
      case result {
        Ok(AudioChunk(data)) -> {
          #(
            Model(..model, status_message: "Streaming audio..."),
            effect.from(fn(_) { play_chunk(data) }),
          )
        }
        Ok(StreamDone(_)) -> {
          // Stream finished, maybe reset UI state
          #(Model(..model, status_message: "Done"), effect.none())
        }
        Error(_) -> #(model, effect.none())
      }
    }

    WsEvent(ws.OnBinaryMessage(_)) -> {
      #(model, effect.none())
    }

    WsEvent(ws.OnClose(_reason)) -> {
      #(
        Model(
          ..model,
          ws: None,
          is_connected: False,
          status_message: "Connection lost",
        ),
        effect.none(),
      )
    }

    UpdateInput(text) -> {
      #(Model(..model, input_text: text), effect.none())
    }

    GenerateAudio -> {
      case model.ws, model.is_connected {
        Some(socket), True -> {
          // Construct the strict Cartesia TTS streaming payload

          let payload =
            json.object([
              #("context_id", json.string("session-context-1")),
              #("model_id", json.string("sonic-3")),
              #("transcript", json.string(model.input_text)),
              #(
                "voice",
                echo build_voice_payload(
                  "a0e99841-438c-4a64-b679-ae501e7d6091",
                  0.8,
                  "calm:highest",
                ),
              ),
              #(
                "output_format",
                json.object([
                  #("container", json.string("raw")),
                  #("encoding", json.string("pcm_f32le")),
                  #("sample_rate", json.int(24_000)),
                ]),
              ),
            ])

          let json_payload = json.to_string(payload)
          #(
            Model(..model, status_message: "Generating..."),
            effect.batch([
              effect.from(fn(_) {
                mount_visualizer("p5-container", get_analyser())
              }),
              ws.send(socket, json_payload),
            ]),
          )
        }
        _, _ -> {
          #(
            Model(..model, status_message: "Socket not connected."),
            effect.none(),
          )
        }
      }
    }
  }
}
// // --- VIEW ---

// pub fn view(model: Model) -> Element(Msg) {
//   html.div(
//     [
//       attribute.class(
//         "min-h-screen bg-slate-900 text-slate-200 flex flex-col items-center justify-center p-6 transition-colors duration-500",
//       ),
//     ],
//     [
//       html.div(
//         [
//           attribute.class(
//             "max-w-md w-full bg-slate-800 rounded-xl shadow-2xl p-8 border border-slate-700/50 backdrop-blur-sm",
//           ),
//         ],
//         [
//           html.h1(
//             [
//               attribute.class(
//                 "text-2xl font-semibold mb-6 text-emerald-400 tracking-wide",
//               ),
//             ],
//             [html.text("TTS Streamer")],
//           ),
//           html.div([attribute.class("mb-4 flex items-center space-x-3")], [
//             html.div(
//               [
//                 attribute.class(
//                   "w-3 h-3 rounded-full transition-all duration-300 ",
//                   // <> if model.is_connected { "bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.8)]" } 
//                 //    else { "bg-rose-500/80" },
//                 ),
//               ],
//               [],
//             ),
//             html.span([attribute.class("text-sm text-slate-400 font-medium")], [
//               html.text(model.status_message),
//             ]),
//           ]),
//           html.textarea(
//             [
//               attribute.class(
//                 "w-full h-32 bg-slate-900/80 border border-slate-700 rounded-lg p-4 text-slate-300 focus:outline-none focus:ring-2 focus:ring-emerald-500/30 resize-none mb-6 placeholder-slate-600 transition-all",
//               ),
//               event.on_input(UpdateInput),
//             ],
//             model.input_text,
//           ),
//           html.div(
//             [
//               attribute.id("p5-container"),
//               attribute.class("w-64 h-64 rounded-full overflow-hidden"),
//             ],
//             [],
//           ),
//           html.button(
//             [
//               attribute.class(
//                 "w-full py-3 rounded-lg font-medium transition-all duration-300 ",
//                 // <> if model.is_connected {
//               //   "bg-emerald-600 hover:bg-emerald-500 text-slate-50 shadow-lg shadow-emerald-900/20 active:scale-[0.98]"
//               // } else {
//               //   "bg-slate-700/50 text-slate-500 cursor-not-allowed"
//               // },
//               ),
//               event.on_click(GenerateAudio),
//               attribute.disabled(!model.is_connected),
//             ],
//             [html.text("Synthesize")],
//           ),
//         ],
//       ),
//     ],
//   )
// }

// pub fn main() {
//   let app = lustre.application(init, update, view)
//   let assert Ok(_) = lustre.start(app, "#app", Nil)
// }
