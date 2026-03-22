import gleam/dynamic/decode
import gleam/function
import gleam/json
import gleam/option.{type Option, None, Some}
import lustre/effect
import rsvp

pub type AgentResponse {
  AgentResponse(thread_id: String, reply: String, synth_ready: Bool)
}

pub type TaskStatus {
  Idle
  Processing
  Completed(answer: String, transcript: String)
  Failed(error: String)
}

fn decode_agent_response() -> decode.Decoder(AgentResponse) {
  use thread_id <- decode.field("thread_id", decode.string)
  use reply <- decode.field("reply", decode.string)
  use synth_ready <- decode.field("synth_ready", decode.bool)
  decode.success(AgentResponse(thread_id:, reply:, synth_ready:))
}

fn status_decoder() -> decode.Decoder(TaskStatus) {
  use status <- decode.field("status", decode.string)
  use answer <- decode.field("answer", decode.string)
  use transcript <- decode.field("transcript", decode.string)

  case status {
    "working" -> decode.success(Processing)
    "success" -> decode.success(Completed(answer:, transcript:))
    "failed" -> decode.success(Failed("Synthesis failed reflection"))
    _ -> decode.success(Idle)
  }
}

pub fn send_message(
  content: String,
  thread_id: Option(String),
) -> effect.Effect(Result(AgentResponse, rsvp.Error)) {
  let payload = case thread_id {
    Some(sid) ->
      json.object([
        #("message", json.string(content)),
        #("thread_id", json.string(sid)),
      ])

    None -> json.object([#("message", json.string(content))])
  }

  let url = "http://localhost:8000/v1/mindfulness/chat"
  let handler = rsvp.expect_json(decode_agent_response(), function.identity)

  rsvp.post(url, payload, handler)
}

pub fn call_synthesis_api(thread_id: String) {
  let url = "http://localhost:8000/v1/mindfulness/synthesis/" <> thread_id
  let handler = rsvp.expect_json(status_decoder(), function.identity)

  rsvp.get(url, handler)
}
