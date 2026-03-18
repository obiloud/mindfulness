import lustre/attribute
import lustre/element
import lustre/element/html
import lustre/event
import mork
import mork/to_lustre

pub fn view_meditation_screen(
  message: String,
  audio_ctrl: msg,
  hide: msg,
) -> element.Element(msg) {
  html.div(
    [
      attribute.class(
        "min-h-screen flex flex-col items-center justify-center gap-6 p-6 bg-tactile-nature text-text-base transition-colors duration-500",
      ),
    ],
    [
      // Centered message bubble
      html.div(
        [
          attribute.class("max-w-[85%] px-5 py-4 text-bubble-agent-text"),
        ],
        message
          |> mork.parse
          |> to_lustre.to_lustre,
      ),

      html.div([attribute.class("my-8 relative")], [
        // The container p5 will attach to
        html.div(
          [
            attribute.id("p5-container"),
            attribute.class(
              "w-64 h-64 rounded-full overflow-hidden absolute transform left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2",
            ),
          ],
          [],
        ),
        html.div(
          [
            attribute.class(
              "text-warm-sand w-[200px] h-[200px] text-[12em] rounded-full transition-all duration-300 "
              <> "ring-warm-sand ring-4 flex items-center justify-center "
              <> "flex align-center justify-center ",
            ),
          ],
          [
            html.p(
              [
                attribute.class(
                  "w-[180px] h-[180px] rounded-full flex align-center justify-center border",
                ),
              ],
              [
                html.i(
                  [
                    attribute.class(
                      "icon-pulse-lotus inline-flex transfom -translate-y-[10px]",
                    ),
                  ],
                  [],
                ),
              ],
            ),
          ],
        ),
      ]),

      // Large play/pause button (centered below message)
      html.div(
        [
          attribute.class("flex justify-center"),
        ],
        [
          html.button(
            [
              event.on_click(audio_ctrl),
              attribute.class(
                "h-16 w-16 text-8xl rounded-full flex items-center justify-center overflow-hidden text-text-base cursor-pointer transition-all duration-300 shadow-lg hover:shadow-xl",
              ),
            ],
            [html.i([attribute.class("icon-play inline-flex")], [])],
          ),
        ],
      ),

      html.div([event.on_click(hide)], [
        html.button([], [element.text("Close")]),
      ]),
    ],
  )
}
