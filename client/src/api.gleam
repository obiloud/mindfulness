import gleam/dynamic/decode
import gleam/function
import gleam/http
import gleam/http/request
import gleam/json
import gleam/option.{type Option, None, Some}
import lustre/effect
import rsvp

pub type User {
  User(email: String, password: String)
}

pub type AuthResponse {
  AuthResponse(access_token: String, token_type: String)
}

fn decode_auth_response() -> decode.Decoder(AuthResponse) {
  use access_token <- decode.field("access_token", decode.string)
  use token_type <- decode.field("token_type", decode.string)
  decode.success(AuthResponse(access_token:, token_type:))
}

pub type AgentResponse {
  AgentResponse(
    thread_id: String,
    user_id: String,
    reply: String,
    answer: Option(String),
    transcript: Option(String),
  )
}

fn decode_agent_response() -> decode.Decoder(AgentResponse) {
  use thread_id <- decode.field("thread_id", decode.string)
  use user_id <- decode.field("user_id", decode.string)
  use reply <- decode.field("reply", decode.string)
  use answer <- decode.field("answer", decode.optional(decode.string))
  use transcript <- decode.field("transcript", decode.optional(decode.string))
  decode.success(AgentResponse(
    thread_id:,
    user_id:,
    reply:,
    answer:,
    transcript:,
  ))
}

pub fn send_message(
  content: String,
  thread_id: Option(String),
  access_token: String,
) -> effect.Effect(Result(AgentResponse, rsvp.Error)) {
  let payload = case thread_id {
    Some(sid) ->
      json.object([
        #("message", json.string(content)),
        #("thread_id", json.string(sid)),
      ])

    None -> json.object([#("message", json.string(content))])
  }

  let handler = rsvp.expect_json(decode_agent_response(), function.identity)

  request.new()
  |> request.set_method(http.Post)
  |> request.set_header("content-type", "application/json")
  |> request.set_header("authorization", "Bearer " <> access_token)
  |> request.set_scheme(http.Http)
  |> request.set_host("localhost")
  |> request.set_port(8000)
  |> request.set_path("/v1/mindfulness/chat")
  |> request.set_body(json.to_string(payload))
  |> rsvp.send(handler)
}

pub fn register(
  email: String,
  password: String,
) -> effect.Effect(Result(AuthResponse, rsvp.Error)) {
  let payload =
    json.object([
      #("email", json.string(email)),
      #("password", json.string(password)),
    ])

  let url = "http://localhost:8000/auth/register"
  let handler = rsvp.expect_json(decode_auth_response(), function.identity)

  rsvp.post(url, payload, handler)
}

pub fn login(
  email: String,
  password: String,
) -> effect.Effect(Result(AuthResponse, rsvp.Error)) {
  let payload =
    json.object([
      #("email", json.string(email)),
      #("password", json.string(password)),
    ])

  let url = "http://localhost:8000/auth/login"
  let handler = rsvp.expect_json(decode_auth_response(), function.identity)

  rsvp.post(url, payload, handler)
}
