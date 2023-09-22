# -*- coding: utf-8 -*-

# This file contains function linkcode_resolve, based on
# https://github.com/numpy/numpy/blob/main/doc/source/conf.py,
# which is licensed under the BSD 3-Clause "New" or "Revised"
# license: ./licenses/numpy.rst

import configparser
import os
import sys
import subprocess
import inspect
import pkg_resources

sdk_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, sdk_directory)

# -- Project information - these are special values used by sphinx. -------

from dynex import __version__ as version
from dynex import __version__ as release

setup_cfg = configparser.ConfigParser()
setup_cfg.read(os.path.join(sdk_directory, 'setup.cfg'))

author = setup_cfg['metadata']['author']
copyright = setup_cfg['metadata']['author']

project = 'Dynex SDK Documentation'

# Also add our own 'special value', the minimum supported Python version
rst_prolog = f" .. |python_requires| replace:: {setup_cfg['options']['python_requires']}"

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.linkcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.ifconfig',
    'breathe',
    'sphinx_panels',
    'reno.sphinxext',
    'sphinx_copybutton',
]

autosummary_generate = True

source_suffix = ['.rst', '.md']

root_doc = 'index'  # before Sphinx 4.0, named master_doc

add_module_names = False

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

linkcheck_ignore = [r'.clang-format',                    # would need symlink
                    r'setup.cfg',                        # would need symlink (for dimod)
                    r'https://cloud.dwavesys.com/leap',  # redirects, many checks
                    r'https://scipy.org',                # ignores robots
                    r'https://epubs.siam.org',           # ignores robots since Feb 2023
                    r'LICENSE',                          # would need symlink, checked by submodule
                    r'CONTRIBUTING',                     # would need symlink, checked by submodule
                    ]

pygments_style = 'sphinx'

todo_include_todos = True

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

doctest_global_setup = """

import dynex
import dimod

# -- Options for HTML output ----------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_logo = ""

html_theme_options = {
    "github_url": "https://github.com/dynexcoin/DynexSDK",
    "external_links": [
        {
            "url": "https://github.com/dynexcoin/DynexSDK/wiki",
            "name": "Dynex SDK Wiki",
        },
        {
            "url": "https://dynexcoin.org",
            "name": "Dynex Website",
        },
        {
            "url": "https://blockexplorer.dynexcoin.org",
            "name": "Dynex Explorer",
        },
    ],
    
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"]  # remove ads
}
html_static_path = ['_static']


def setup(app):
   app.add_css_file('theme_overrides.css')
   app.add_css_file('cookie_notice.css')
   app.add_js_file('cookie_notice.js')
   app.add_config_value('target', 'sdk', 'env')


# -- Panels ---------------------------------------------------------------
panels_add_bootstrap_css = False

# -- Intersphinx ----------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('http://numpy.org/doc/stable/', None),
    }


# -- Linkcode -------------------------------------------------------------
github_map = {'Dynex SDK': 'dynex',
              'PyTorch': 'dynex_pytorch',
              'QBoost':  'dynex_qboost',
              'Scikit Learn': 'dynex_scikit_plugin',
              'QRBM': 'QRBM',
              'DynexQRBM': 'DynexQRBM',
              'HybridQRBM': 'HybridQRBM',
              'CFQIQRBM': 'CFQIQRBM',
              }

reqs = pkg_resources.get_distribution('DynexSDK').requires(extras=['all'])
pkgs = [pkg_resources.get_distribution(req) for req in reqs]
versions = {pkg.project_name: pkg.version for pkg in pkgs}

def linkcode_resolve(domain, info):
    """
    Find the URL of the GitHub source for DynexSDK objects.
    """
    # Based on https://github.com/numpy/numpy/blob/main/doc/source/conf.py
    # Updated to work on multiple submodules and fall back to next-level
    # module for objects such as properties

    if domain != 'py':
        return None

    obj={}
    obj_inx = 0
    obj[obj_inx] = sys.modules.get(info['module'])
    for part in info['fullname'].split('.'):
        obj_inx += 1
        try:
            obj[obj_inx] = getattr(obj[obj_inx - 1], part)
        except Exception:
            pass

    # strip decorators, which would resolve to the source of the decorator
    # https://bugs.python.org/issue34305
    for i in range(len(obj)):
        obj[i] = inspect.unwrap(obj[i])

    fn = None
    for i in range(len(obj)-1, -1, -1):
        try:
            fn = inspect.getsourcefile(obj[i])
            if fn:
                obj_inx = i
                break
        except:
            pass

    linespec = ""
    try:
        source, lineno = inspect.getsourcelines(obj[obj_inx])
        if obj_inx != 0:
            linespec = "#L%d" % (lineno)
    except Exception:
        linespec = ""

    if not fn or not "site-packages" in fn:
        return None

    if ".egg" in fn:
        fn = fn.replace(fn[:fn.index("egg")+len("egg")], "")
    else:
        fn = fn.replace(fn[:fn.index("site-packages")+len("site-packages")], "")

    repo = fn.split("/")[1] if  \
        (fn.split("/")[1] != "dwave") \
        else fn.split("/")[2]

    pm_module = github_map[repo]
    pm_ver = versions[github_map[repo]]
    fn = "https://github.com/dynexcoin/{}/blob/{}{}".format(pm_module, pm_ver, fn)

    return fn + linespec
