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

html_theme = "pydata_sphinx_theme"
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
html_css_files = ["custom.css"]
# The gallery images are produced by docs/make_gallery.py, which runs the same
# public functions the pages describe. Regenerate them when the plotting API
# changes; a figure that no longer matches its code is worse than no figure.

autosummary_generate = True
autodoc_typehints = "description"
nb_execution_mode = "off"

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# HUMAN: the left sidebar is useless on every page. Removed globally. The
# top navbar carries the sections and the right-hand page TOC carries the
# within-page headings, so nothing is lost and the content gets the width.
html_sidebars = {"**": []}

html_theme_options = {
    "show_prev_next": True,
    "navbar_align": "left",
    "github_url": "https://github.com/yuvrajiro/lairnet",
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
        "text": "LAIR-Net",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    # Every section in the top bar; no "More" dropdown to hide things behind.
    "header_links_before_dropdown": 8,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/lairnet/",
            "icon": "fa-brands fa-python",
        },
    ],
    "show_toc_level": 2,
}
