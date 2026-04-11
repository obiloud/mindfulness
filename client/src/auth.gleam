// --- AUTHENTICATION MODULE ---

import api.{type AuthResponse, login, register}
import gleam/dynamic/decode
import gleam/json
import gleam/option.{type Option, None, Some}
import local_storage
import lustre/effect.{type Effect}
import rsvp

// --- TYPES ---

pub type AuthScreen {
  LoginScreen
  RegisterScreen
}

pub type AuthState {
  AuthState(
    email: String,
    password: String,
    confirm_password: String,
    auth_screen: Option(AuthScreen),
    validation_error: Option(String),
    access_token: Option(String),
    refresh_token: Option(String),
  )
}

pub type AuthMsg {
  EmailChanged(String)
  PasswordChanged(String)
  ConfirmPasswordChanged(String)
  AttemptLogin
  AttemptRegister
  OnResult(Result(AuthResponse, rsvp.Error))
  ToggleAuthScreen(AuthScreen)
  AuthSuccess(String, String)
  RefreshStarted
  RefreshCompleted(Result(AuthResponse, rsvp.Error))
  RefreshSuccess(String, String)
  LogoutRequested
  LogoutSuccess
}

// --- AUTHENTICATION LOGIC ---

pub fn auth_init() -> #(AuthState, Effect(AuthMsg)) {
  #(
    AuthState(
      email: "",
      password: "",
      confirm_password: "",
      auth_screen: Some(LoginScreen),
      validation_error: None,
      access_token: None,
      refresh_token: None,
    ),
    effect.from(fn(dispatch) {
      let access_token = local_storage.get_item("access_token")
      let refresh_token = local_storage.get_item("refresh_token")
      case echo access_token, refresh_token {
        Some(access), Some(refresh) ->
          dispatch(
            OnResult(
              Ok(api.AuthResponse(
                access_token: access,
                refresh_token: refresh,
                token_type: "Bearer",
              )),
            ),
          )
        _, _ -> Nil
      }
    }),
  )
}

pub fn update_auth_state(
  state: AuthState,
  msg: AuthMsg,
) -> #(AuthState, Effect(AuthMsg)) {
  case msg {
    EmailChanged(email) -> {
      let new_state = AuthState(..state, email: email)
      #(new_state, effect.none())
    }

    PasswordChanged(password) -> {
      let new_state = AuthState(..state, password: password)
      #(new_state, effect.none())
    }

    ConfirmPasswordChanged(confirm_password) -> {
      let new_state = AuthState(..state, confirm_password: confirm_password)
      #(new_state, effect.none())
    }

    AttemptLogin -> {
      case state.email == "" || state.password == "" {
        True -> {
          let new_state =
            AuthState(
              ..state,
              validation_error: Some("Please fill in all fields"),
            )
          #(new_state, effect.none())
        }
        False -> {
          let new_state = AuthState(..state, email: "", password: "")
          #(new_state, effect.map(login(state.email, state.password), OnResult))
        }
      }
    }

    AttemptRegister -> {
      case state.email == "" || state.password == "" {
        True -> {
          let new_state =
            AuthState(
              ..state,
              validation_error: Some("Please fill in all fields"),
            )
          #(new_state, effect.none())
        }
        False ->
          case state.password != state.confirm_password {
            True -> {
              let new_state =
                AuthState(
                  ..state,
                  validation_error: Some("Passwords do not match"),
                )
              #(new_state, effect.none())
            }
            False -> {
              let new_state = AuthState(..state, email: "", password: "")
              #(
                new_state,
                effect.map(register(state.email, state.password), OnResult),
              )
            }
          }
      }
    }

    ToggleAuthScreen(screen) -> {
      #(
        AuthState(..state, auth_screen: Some(screen), validation_error: None),
        effect.none(),
      )
    }

    OnResult(Ok(response)) -> #(
      AuthState(..state, auth_screen: None, validation_error: None),
      effect.from(fn(dispatch) {
        let success =
          local_storage.set_item("access_token", response.access_token)
        let refresh_success =
          local_storage.set_item("refresh_token", response.refresh_token)
        case success, refresh_success {
          True, True ->
            dispatch(AuthSuccess(response.access_token, response.refresh_token))
          _, _ -> Nil
        }
      }),
    )

    OnResult(Error(rsvp.HttpError(response))) -> {
      let detail_decoder = {
        use detail <- decode.field("detail", decode.string)
        decode.success(detail)
      }
      let result = json.parse(response.body, detail_decoder)
      #(
        AuthState(..state, validation_error: option.from_result(result)),
        effect.none(),
      )
    }

    OnResult(Error(_)) -> #(
      AuthState(..state, validation_error: Some("Something went wrong")),
      effect.none(),
    )

    AuthSuccess(_, _) -> #(state, effect.none())

    RefreshStarted -> #(
      AuthState(..state, access_token: None, refresh_token: None),
      effect.from(fn(_) {
        // Clear tokens during refresh
        local_storage.remove_item("access_token")
        local_storage.remove_item("refresh_token")
        Nil
      }),
    )

    RefreshCompleted(Ok(response)) -> #(
      AuthState(..state, auth_screen: None, validation_error: None),
      effect.from(fn(dispatch) {
        let success =
          local_storage.set_item("access_token", response.access_token)
        let refresh_success =
          local_storage.set_item("refresh_token", response.refresh_token)
        case echo success, refresh_success {
          True, True ->
            dispatch(RefreshSuccess(
              response.access_token,
              response.refresh_token,
            ))
          _, _ -> Nil
        }
      }),
    )

    RefreshCompleted(_) -> #(
      AuthState(
        ..state,
        access_token: None,
        refresh_token: None,
        auth_screen: Some(LoginScreen),
      ),
      effect.batch([
        effect.from(fn(_) {
          local_storage.remove_item("access_token")
          local_storage.remove_item("refresh_token")
          Nil
        }),
        effect.from(fn(dispatch) { dispatch(LogoutSuccess) }),
      ]),
    )

    RefreshSuccess(_, _) -> #(state, effect.none())

    LogoutRequested -> #(
      AuthState(
        ..state,
        access_token: None,
        refresh_token: None,
        auth_screen: Some(LoginScreen),
      ),
      effect.batch([
        effect.from(fn(_) {
          local_storage.remove_item("access_token")
          local_storage.remove_item("refresh_token")
          Nil
        }),
        effect.from(fn(dispatch) { dispatch(LogoutSuccess) }),
      ]),
    )

    LogoutSuccess -> #(state, effect.none())
  }
}
