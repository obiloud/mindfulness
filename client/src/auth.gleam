// --- AUTHENTICATION MODULE ---

import api.{type AuthResponse, login, register}
import gleam/option.{type Option, None, Some}
import lustre/attribute
import lustre/effect.{type Effect}
import lustre/element.{type Element}
import lustre/element/html
import lustre/event
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
    auth_screen: Option(AuthScreen),
    validation_error: Option(String),
  )
}

pub type AuthMsg {
  EmailChanged(String)
  PasswordChanged(String)
  AttemptLogin
  AttemptRegister
  OnResult(Result(AuthResponse, rsvp.Error))
  ToggleAuthScreen(AuthScreen)
  AuthSuccess(String)
}

// --- AUTHENTICATION LOGIC ---

pub fn auth_init() -> #(AuthState, Effect(AuthMsg)) {
  #(
    AuthState(
      email: "",
      password: "",
      auth_screen: None,
      validation_error: None,
    ),
    effect.none(),
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
        False -> {
          let new_state = AuthState(..state, email: "", password: "")
          #(
            new_state,
            effect.map(register(state.email, state.password), OnResult),
          )
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
      effect.from(fn(dispatch) { dispatch(AuthSuccess(response.access_token)) }),
    )

    OnResult(Error(_)) -> #(
      AuthState(..state, validation_error: Some("Wrong email or password")),
      effect.none(),
    )

    AuthSuccess(_) -> #(state, effect.none())
  }
}

// --- AUTHENTICATION VIEW ---

pub fn view_auth_screen(state: AuthState) -> Element(AuthMsg) {
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
          case state.auth_screen {
            Some(LoginScreen) -> {
              html.div(
                [
                  attribute.class(
                    "min-h-screen bg-bg-main font-body text-text-base",
                  ),
                ],
                [
                  html.div(
                    [
                      attribute.class(
                        "max-w-md w-full bg-white dark:bg-gray-800 p-8 shadow-lg",
                      ),
                    ],
                    [
                      html.h1(
                        [
                          attribute.class(
                            "text-2xl font-bold text-center mb-6 text-deep-moss",
                          ),
                        ],
                        [element.text("Login to PULSE LOTUS")],
                      ),
                      html.input([
                        attribute.type_("email"),
                        attribute.class(
                          "w-full px-4 py-2 border border-gray-300",
                        ),
                        event.on_input(EmailChanged),
                        attribute.value(state.email),
                      ]),
                      html.input([
                        attribute.type_("password"),
                        attribute.class(
                          "w-full px-4 py-2 border border-gray-300",
                        ),
                        event.on_input(PasswordChanged),
                        attribute.value(state.password),
                      ]),
                      case state.validation_error {
                        Some(validation_error) ->
                          html.p(
                            [attribute.class("text-red-500 text-sm mb-4")],
                            [
                              element.text(validation_error),
                            ],
                          )
                        None -> html.text("")
                      },
                      html.button(
                        [
                          event.on_click(AttemptLogin),
                          attribute.class("px-6 py-2 bg-neo-mint"),
                        ],
                        [element.text("Submit")],
                      ),
                      html.p([attribute.class("flex gap-2 my-2")], [
                        html.span(
                          [
                            attribute.class(
                              "text-sm text-gray-600 dark:text-gray-400",
                            ),
                          ],
                          [element.text("Don't have an account?")],
                        ),
                        html.button(
                          [
                            event.on_click(ToggleAuthScreen(RegisterScreen)),
                            attribute.class(
                              "text-neo-mint hover:underline text-sm cursor-pointer",
                            ),
                          ],
                          [element.text("Register")],
                        ),
                      ]),
                    ],
                  ),
                ],
              )
            }

            Some(RegisterScreen) -> {
              html.div(
                [
                  attribute.class(
                    "min-h-screen bg-bg-main font-body text-text-base",
                  ),
                ],
                [
                  html.div(
                    [
                      attribute.class(
                        "max-w-md w-full bg-white dark:bg-gray-800 p-8 shadow-lg",
                      ),
                    ],
                    [
                      html.h1(
                        [
                          attribute.class(
                            "text-2xl font-bold text-center mb-6 text-deep-moss",
                          ),
                        ],
                        [element.text("Create Account")],
                      ),
                      html.input([
                        attribute.type_("email"),
                        attribute.class(
                          "w-full px-4 py-2 border border-gray-300",
                        ),
                        event.on_input(EmailChanged),
                        attribute.value(state.email),
                      ]),
                      html.input([
                        attribute.type_("password"),
                        attribute.class(
                          "w-full px-4 py-2 border border-gray-300",
                        ),
                        event.on_input(PasswordChanged),
                        attribute.value(state.password),
                      ]),
                      case state.validation_error {
                        Some(validation_error) ->
                          html.p(
                            [attribute.class("text-red-500 text-sm mb-4")],
                            [
                              element.text(validation_error),
                            ],
                          )
                        None -> html.text("")
                      },
                      html.button(
                        [
                          event.on_click(AttemptRegister),
                          attribute.class("px-6 py-2 bg-neo-mint"),
                        ],
                        [element.text("Submit")],
                      ),
                      html.p([attribute.class("flex gap-2 my-2")], [
                        html.span(
                          [
                            attribute.class(
                              "text-sm text-gray-600 dark:text-gray-400",
                            ),
                          ],
                          [element.text("Already have an account?")],
                        ),
                        html.button(
                          [
                            event.on_click(ToggleAuthScreen(LoginScreen)),
                            attribute.class(
                              "text-neo-mint hover:underline text-sm cursor-pointer",
                            ),
                          ],
                          [element.text("Login")],
                        ),
                      ]),
                    ],
                  ),
                ],
              )
            }

            None -> html.text("")
          },
        ],
      ),
    ],
  )
}
