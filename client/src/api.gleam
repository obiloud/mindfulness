import gleam/dynamic/decode
import gleam/function
import gleam/json
import gleam/option.{type Option, None, Some}
import lustre/effect
import rsvp

pub type AgentResponse {
  AgentResponse(
    session_id: String,
    message: String,
    answer: Option(String),
    transcript: Option(String),
  )
}

fn decode_agent_response() -> decode.Decoder(AgentResponse) {
  use session_id <- decode.field("session_id", decode.string)
  use message <- decode.field("message", decode.string)
  use answer <- decode.field("answer", decode.optional(decode.string))
  use transcript <- decode.field("transcript", decode.optional(decode.string))
  decode.success(AgentResponse(session_id:, message:, answer:, transcript:))
}

pub fn send_message(
  content: String,
  session_id: Option(String),
) -> effect.Effect(Result(AgentResponse, rsvp.Error)) {
  let payload = case session_id {
    Some(sid) ->
      json.object([
        #("query", json.string(content)),
        #("session_id", json.string(sid)),
      ])

    None -> json.object([#("query", json.string(content))])
  }

  let url = "http://localhost:8000/v1/mindfulness/session"
  let handler = rsvp.expect_json(decode_agent_response(), function.identity)

  rsvp.post(url, payload, handler)
}
