import lustre/element/html
import lustre/attribute
import lustre/event
import lustre/element

pub type Theme {
  Light
  Dark
  System
}


pub fn view_theme_toggle(current_theme: Theme, set_theme: fn(Theme) -> msg) -> element.Element(msg) {
  let base_btn_class = "px-4 py-1 text-xs font-medium rounded-full border transition-all duration-300 "

  html.div([attribute.class("flex gap-2 justify-center mt-2 mb-4")], [
    html.button(
      [
        event.on_click(set_theme(Light)),
        attribute.class(base_btn_class <> case current_theme {
          Light -> "bg-neo-mint text-deep-moss border-neo-mint shadow-sm"
          _ -> "bg-transparent text-charcoal/50 border-charcoal/20 hover:border-neo-mint/50"
        }),
      ],
      [element.text("Light")],
    ),
    html.button(
      [
        event.on_click(set_theme(Dark)),
        attribute.class(base_btn_class <> case current_theme {
          Dark -> "bg-neo-mint text-deep-moss border-neo-mint shadow-sm"
          _ -> "bg-transparent text-charcoal/50 border-charcoal/20 hover:border-neo-mint/50"
        }),
      ],
      [element.text("Dark")],
    ),
    html.button(
      [
        event.on_click(set_theme(System)),
        attribute.class(base_btn_class <> case current_theme {
          System -> "bg-neo-mint text-deep-moss border-neo-mint shadow-sm"
          _ -> "bg-transparent text-charcoal/50 border-charcoal/20 hover:border-neo-mint/50"
        }),
      ],
      [element.text("System")],
    ),
  ])
}