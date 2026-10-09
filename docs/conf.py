project = "LAIR-Net"
copyright = "2026"
author = "LAIR-Net contributors"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "furo"
html_title = "LAIR-Net"
html_static_path = ["_static"]
# Served from GitHub Pages at a custom domain. CNAME is what tells Pages
# which domain this site answers on, and .nojekyll stops any Jekyll pass
# from dropping the _static directory Sphinx writes. Both have to sit at
# the root of the built site, which is what html_extra_path does.
html_baseurl = "https://lairnet.statml.in/"
html_extra_path = ["CNAME", ".nojekyll"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"

# Typography carries this design, so the faces are the one external resource
# the site loads: Instrument Serif for display, Inter for text, JetBrains Mono
# for code. custom.css is listed last so it wins over the theme's own sheet.
html_css_files = [
    ("https://fonts.googleapis.com/css2"
     "?family=Instrument+Serif:ital@0;1"
     "&family=Inter:wght@400;500;600;700"
     "&family=JetBrains+Mono:wght@400;500"
     "&display=swap"),
    "custom.css",
]

# The gallery images are produced by docs/make_gallery.py, which runs the same
# public functions the pages describe. Regenerate them when the plotting API
# changes; a figure that no longer matches its code is worse than no figure.

autosummary_generate = True
autodoc_typehints = "description"
nb_execution_mode = "off"

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

_GITHUB_SVG = (
    '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
    'viewBox="0 0 16 16" height="1em" width="1em">'
    '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 '
    '5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-'
    '2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 '
    '1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-'
    '3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2'
    '.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82'
    '.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95'
    '.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 '
    '8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
)

# Furo derives every colour from these two blocks, so setting the brand to
# black and near-white is what makes the site monochrome in both modes. No
# accent colour is defined anywhere, by design.
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#111111",
        "color-brand-content": "#111111",
        "color-brand-visited": "#111111",
    },
    "dark_css_variables": {
        "color-brand-primary": "#f2f2f2",
        "color-brand-content": "#f2f2f2",
        "color-brand-visited": "#f2f2f2",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/yuvrajiro/lairnet",
            "html": _GITHUB_SVG,
            "class": "",
        },
    ],
}
