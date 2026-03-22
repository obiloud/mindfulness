import gleam/dynamic/decode
import gleam/function
import gleam/json
import gleam/option.{type Option, None, Some}
import lustre/effect
import rsvp

pub type AgentResponse {
  AgentResponse(
    thread_id: String,
    reply: String,
    answer: Option(String),
    transcript: Option(String),
  )
}

fn decode_agent_response() -> decode.Decoder(AgentResponse) {
  use thread_id <- decode.field("thread_id", decode.string)
  use reply <- decode.field("reply", decode.string)
  use answer <- decode.field("answer", decode.optional(decode.string))
  use transcript <- decode.field("transcript", decode.optional(decode.string))
  decode.success(AgentResponse(thread_id:, reply:, answer:, transcript:))
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
