import auth.{
  type AuthMsg, type AuthState, AttemptLogin, AttemptRegister,
  ConfirmPasswordChanged, EmailChanged, LoginScreen, PasswordChanged,
  RegisterScreen, ToggleAuthScreen,
}
import gleam/dynamic/decode
import gleam/list
import gleam/option.{None, Some}
import gleam/string
import lustre/attribute
import lustre/element.{type Element}
import lustre/element/html
import lustre/event

pub fn view_auth_screen(state: AuthState) -> Element(AuthMsg) {
  html.div(
    [
      attribute.class(
        "min-h-screen flex items-center justify-center bg-[#F8FAF7]",
      ),
    ],
    [
      html.div(
        [
          attribute.class(
            "bg-white p-8 rounded-2xl shadow-lg w-full max-w-md border border-[#E8EDEA] transition-all duration-500 ease-in-out",
          ),
        ],
        [
          html.h1(
            [
              attribute.class(
                "text-2xl font-medium text-[#2C3E36] mb-6 text-center tracking-tight",
              ),
            ],
            [element.text("Welcome Back")],
          ),

          // Form fields go here
          login_form(state),

          // Social Auth Placeholder
          social_auth_section(),
        ],
      ),
    ],
  )
}

fn login_form(state: AuthState) {
  case state.auth_screen {
    Some(LoginScreen) -> {
      html.div(
        [
          attribute.class("w-full"),
        ],
        [
          html.div(
            [
              attribute.class("w-full space-y-5"),
            ],
            [
              // Email field with label
              html.div([attribute.class("space-y-1")], [
                html.label(
                  [
                    attribute.class(
                      "block text-sm font-medium text-[#2C3E36] mb-1",
                    ),
                  ],
                  [element.text("Email Address")],
                ),
                html.input([
                  attribute.type_("email"),
                  attribute.class(
                    "w-full px-4 py-2.5 border border-[#E8EDEA] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6B8E78] focus:border-transparent transition-all duration-200 text-[#2C3E36] placeholder-[#9CA3AF]",
                  ),
                  event.on_input(EmailChanged),
                  event.advanced("keydown", on_enter_key(AttemptLogin)),
                  attribute.value(state.email),
                  attribute.placeholder("Enter your email"),
                ]),
              ]),

              // Password field with label
              html.div([attribute.class("space-y-1")], [
                html.label(
                  [
                    attribute.class(
                      "block text-sm font-medium text-[#2C3E36] mb-1",
                    ),
                  ],
                  [element.text("Password")],
                ),
                html.input([
                  attribute.type_("password"),
                  attribute.class(
                    "w-full px-4 py-2.5 border border-[#E8EDEA] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6B8E78] focus:border-transparent transition-all duration-200 text-[#2C3E36] placeholder-[#9CA3AF]",
                  ),
                  event.on_input(PasswordChanged),
                  event.advanced("keydown", on_enter_key(AttemptLogin)),
                  attribute.value(state.password),
                  attribute.placeholder("Enter your password"),
                ]),
              ]),

              // Validation error message
              case state.validation_error {
                Some(validation_error) ->
                  html.p([attribute.class("text-[#C45A5A] text-sm mt-2")], [
                    element.text(validation_error),
                  ])
                None -> html.text("")
              },

              // Sign in button
              html.button(
                [
                  event.on_click(AttemptLogin),
                  attribute.class(
                    "w-full py-2.5 px-4 bg-[#6B8E78] hover:bg-[#5A7A66] text-white font-medium rounded-lg transition-colors duration-200 shadow-sm hover:shadow-md",
                  ),
                ],
                [element.text("Sign In")],
              ),

              // Forgot password and register link
              html.div([attribute.class("flex flex-col gap-2 my-4")], [
                html.a(
                  [
                    attribute.class(
                      "text-sm text-[#6B8E78] hover:underline cursor-pointer",
                    ),
                  ],
                  [element.text("Forgot Password?")],
                ),
                html.p([attribute.class("flex gap-2 my-2")], [
                  html.span(
                    [
                      attribute.class("text-sm text-[#5A6E65]"),
                    ],
                    [element.text("Don't have an account?")],
                  ),
                  html.button(
                    [
                      event.on_click(ToggleAuthScreen(RegisterScreen)),
                      attribute.class(
                        "text-[#6B8E78] hover:underline text-sm font-medium cursor-pointer",
                      ),
                    ],
                    [element.text("Sign Up")],
                  ),
                ]),
              ]),
            ],
          ),
        ],
      )
    }

    Some(RegisterScreen) -> {
      html.div(
        [
          attribute.class("w-full"),
        ],
        [
          html.div(
            [
              attribute.class("w-full space-y-5"),
            ],
            [
              html.h1(
                [
                  attribute.class(
                    "text-2xl font-medium text-center mb-6 text-[#2C3E36] tracking-tight",
                  ),
                ],
                [element.text("Create Your Account")],
              ),
              // Email field with label
              html.div([attribute.class("space-y-1")], [
                html.label(
                  [
                    attribute.class(
                      "block text-sm font-medium text-[#2C3E36] mb-1",
                    ),
                  ],
                  [element.text("Email Address")],
                ),
                html.input([
                  attribute.type_("email"),

                  attribute.class(
                    "w-full px-4 py-2.5 border border-[#E8EDEA] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6B8E78] focus:border-transparent transition-all duration-200 text-[#2C3E36] placeholder-[#9CA3AF]",
                  ),
                  event.on_input(EmailChanged),
                  event.advanced("keydown", on_enter_key(AttemptRegister)),
                  attribute.value(state.email),
                  attribute.placeholder("Enter your email"),
                ]),
              ]),

              // Password field with label
              html.div([attribute.class("space-y-1")], [
                html.label(
                  [
                    attribute.class(
                      "block text-sm font-medium text-[#2C3E36] mb-1",
                    ),
                  ],
                  [element.text("Password")],
                ),
                html.input([
                  attribute.type_("password"),

                  attribute.class(
                    "w-full px-4 py-2.5 border border-[#E8EDEA] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6B8E78] focus:border-transparent transition-all duration-200 text-[#2C3E36] placeholder-[#9CA3AF]",
                  ),
                  event.on_input(PasswordChanged),
                  event.advanced("keydown", on_enter_key(AttemptRegister)),
                  attribute.value(state.password),
                  attribute.placeholder("Create a password"),
                ]),
              ]),

              // Password strength indicator
              html.div([attribute.class("mt-1")], [
                password_strength_indicator(state.password),
              ]),

              // Confirm password field
              html.div([attribute.class("space-y-1")], [
                html.label(
                  [
                    attribute.class(
                      "block text-sm font-medium text-[#2C3E36] mb-1",
                    ),
                  ],
                  [element.text("Confirm Password")],
                ),
                html.input([
                  attribute.type_("password"),

                  attribute.class(
                    "w-full px-4 py-2.5 border border-[#E8EDEA] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#6B8E78] focus:border-transparent transition-all duration-200 text-[#2C3E36] placeholder-[#9CA3AF]",
                  ),
                  event.on_input(ConfirmPasswordChanged),
                  event.advanced("keydown", on_enter_key(AttemptRegister)),
                  attribute.value(state.confirm_password),
                  attribute.placeholder("Confirm your password"),
                ]),
              ]),

              // Validation error message
              case state.validation_error {
                Some(validation_error) ->
                  html.p([attribute.class("text-[#C45A5A] text-sm mt-2")], [
                    element.text(validation_error),
                  ])
                None -> html.text("")
              },

              // Create account button
              html.button(
                [
                  event.on_click(AttemptRegister),
                  attribute.class(
                    "w-full py-2.5 px-4 bg-[#6B8E78] hover:bg-[#5A7A66] text-white font-medium rounded-lg transition-colors duration-200 shadow-sm hover:shadow-md",
                  ),
                ],
                [element.text("Create Account")],
              ),

              // Already have account link
              html.p([attribute.class("flex gap-2 my-4")], [
                html.span(
                  [
                    attribute.class("text-sm text-[#5A6E65]"),
                  ],
                  [element.text("Already have an account?")],
                ),
                html.button(
                  [
                    event.on_click(ToggleAuthScreen(LoginScreen)),
                    attribute.class(
                      "text-[#6B8E78] hover:underline text-sm font-medium cursor-pointer",
                    ),
                  ],
                  [element.text("Sign In")],
                ),
              ]),
            ],
          ),
        ],
      )
    }

    None -> html.text("")
  }
}

fn password_strength_indicator(password: String) {
  let strength = calculate_password_strength(password)

  html.div([attribute.class("")], [
    // Strength bars
    html.div([attribute.class("flex gap-1 w-full")], [
      strength_bar(strength >= 1),
      strength_bar(strength >= 2),
      strength_bar(strength >= 3),
    ]),

    // Strength text
    html.span(
      [
        attribute.class(
          "text-xs "
          <> case strength {
            0 -> "text-[#9CA3AF]"
            1 -> "text-[#C45A5A]"
            2 -> "text-[#D97706]"
            3 -> "text-[#6B8E78]"
            _ -> "text-[#6B8E78]"
          },
        ),
      ],
      [
        element.text(case strength {
          0 -> "Enter password"
          1 -> "Weak"
          2 -> "Fair"
          3 -> "Good"
          _ -> "Strong"
        }),
      ],
    ),
  ])
}

fn strength_bar(active: Bool) {
  html.div(
    [
      attribute.class(
        "h-1.5 flex-1 rounded-full transition-all duration-200 "
        <> case active {
          True -> "bg-[#6B8E78]"
          False -> "bg-[#E8EDEA]"
        },
      ),
    ],
    [],
  )
}

pub fn contains_any(haystack: String, needles: List(String)) -> Bool {
  list.any(needles, string.contains(haystack, _))
}

fn calculate_password_strength(password: String) -> Int {
  let length = string.length(password)
  let has_uppercase =
    contains_any(password, string.to_graphemes("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
  let has_lowercase =
    contains_any(password, string.to_graphemes("abcdefghijklmnopqrstuvwxyz"))
  let has_numbers = contains_any(password, string.to_graphemes("0123456789"))
  let has_special = contains_any(password, ["!", "@", "#", "$", "%"])

  let base_strength = case length {
    len if len < 1 -> 0
    len if len < 8 -> 1
    len if len < 12 -> 2
    _ -> 3
  }

  let complexity_bonus = case
    has_uppercase,
    has_lowercase,
    has_numbers,
    has_special
  {
    True, True, True, True -> 1
    True, True, True, False -> 1
    True, True, False, True -> 1
    _, _, _, _ -> 0
  }

  base_strength + complexity_bonus
}

fn social_auth_section() {
  html.div([attribute.class("mt-8 pt-6 border-t border-[#E8EDEA]")], [
    html.div([attribute.class("flex flex-col gap-3")], [
      social_button("Continue with Google", "google-icon-path"),
      social_button("Continue with Apple", "apple-icon-path"),
    ]),
  ])
}

fn social_button(label: String, icon: String) {
  html.button(
    [
      attribute.class(
        icon
        <> "flex items-center justify-center gap-2 w-full py-2.5 px-4 border border-[#E8EDEA] rounded-lg text-[#2C3E36] hover:bg-[#F8FAF7] transition-colors duration-200 font-medium",
      ),
    ],
    [element.text(label)],
  )
}

fn on_enter_key(msg: a) -> decode.Decoder(event.Handler(a)) {
  use key <- decode.field("key", decode.string)
  let handler =
    event.handler(dispatch: msg, prevent_default: True, stop_propagation: False)
  case key {
    "Enter" -> decode.success(handler)
    _ -> decode.failure(handler, "on enter")
  }
}
