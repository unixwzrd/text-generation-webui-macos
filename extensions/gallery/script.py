from pathlib import Path

import gradio as gr

from modules.html_generator import get_image_cache
from modules.shared import gradio


def generate_css():
    css = """
      .character-gallery > .gallery {
        margin: 1rem 0;
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        grid-column-gap: 0.4rem;
        grid-row-gap: 1.2rem;
      }

      .character-gallery > .label {
        display: none !important;
      }

      .character-gallery button.gallery-item {
        display: contents;
      }

      .character-container {
        cursor: pointer;
        text-align: center;
        position: relative;
        opacity: 0.85;
      }

      .character-container:hover {
        opacity: 1;
      }

      .character-container .placeholder, .character-container img {
        width: 150px;
        height: 200px;
        background-color: gray;
        object-fit: cover;
        margin: 0 auto;
        border-radius: 1rem;
        border: 3px solid white;
        box-shadow: 3px 3px 6px 0px rgb(0 0 0 / 50%);
      }

      .character-name {
        margin-top: 0.3rem;
        display: block;
        font-size: 1.2rem;
        font-weight: 600;
        overflow-wrap: anywhere;
      }
    """
    return css


def generate_html():
    cards = []
    # Iterate through files in image folder
    for file in sorted(Path("characters").glob("*")):
        if file.suffix in [".json", ".yml", ".yaml"]:
            character = file.stem
            container_html = '<div class="character-container">'
            image_html = "<div class='placeholder'></div>"

            for path in [Path(f"characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
                if path.exists():
                    image_html = f'<img src="file/{get_image_cache(path)}">'
                    break

            container_html += f'{image_html} <span class="character-name">{character}</span>'
            container_html += "</div>"
            cards.append([container_html, character])

    return cards


def select_character(evt: gr.SelectData):
    return (evt.value[1])


def custom_js():
    path_to_js = Path(__file__).parent.resolve() / 'script.js'
    return open(path_to_js, 'r').read()


def ui():
    with gr.Accordion("Character gallery", open=False, elem_id='gallery-extension'):
        update = gr.Button("Refresh")
        gr.HTML(value="<style>" + generate_css() + "</style>")
        gallery = gr.Dataset(components=[gr.HTML(visible=False)],
                             label="",
                             samples=generate_html(),
                             elem_classes=["character-gallery"],
                             samples_per_page=50
                             )
    update.click(generate_html, [], gallery)
    gallery.select(select_character, None, gradio['character_menu'])
