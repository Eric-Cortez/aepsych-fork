import base64
import glob, os, re, traceback, warnings
from re import search

import ipywidgets as widgets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aepsych.plotting import plot_strat, plot_strat_3d
from aepsych.server import AEPsychServer
from aepsych_client import AEPsychClient
from IPython.core.display import HTML as html
from IPython.display import display, HTML
from ipywidgets import Box, Button, HTML, Label, Layout, Text


plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")

connect_out = widgets.Output(layout=Layout(padding="2px", height="150px"))


def clear_logs():
    """
    clears the server.log file data when you start
    """
    full_log_path = os.path.join(os.getcwd(), "logs", "bayes_opt_server.log")
    if os.path.exists(full_log_path):
        open(full_log_path, "w").close()


def clear_databases_dir():
    """
    Deletes all the db file in the database directory
    if directory contains files.
    """
    full_database_path = os.path.join(os.getcwd(), "databases", "*")

    files = glob.glob(full_database_path)
    for f in files:
        os.remove(f)


clear_logs()
clear_databases_dir()


with connect_out:
    server = None
    connect_out.clear_output()

client = None
strat = None
db_file_name = None
database_path = None
new_experiment = False
resume_experiment = False
dim = 0
ip_address = ""

# ---------- Reset Dash ----------

button_reset = widgets.Button(
    description="Reset",
    disabled=False,
    button_style="danger",
    tooltip="submit",
    layout=Layout(align_self="flex-end", width="fit-content", margin="10px"),
)

readme_link = HTML(
    """<p style="margin: 10px;">
Instructions for this dashboard can be found in the
<a style="color: #106ba3;" href="https://github.com/facebookresearch/aepsych/blob/main/Readme.md">
Readme.MD</a>
</p>
"""
)
reset_button_container = widgets.Box(
    [readme_link, button_reset],
    layout=Layout(
        justify_content="space-between", align_items="center", margin="4px 1%"
    ),
)


button_reset.add_class("dash-reset-btn")


# ---------- Style ----------

display(
    html(
        """<style>
*{
    word-break: break-word !important;
}
.readme_link:link { color: #0000EE !important; }
.readme_link:visited { color: #551A8B !important; }
.jp-CodeCell {
    width:100% !important;
    background: whitesmoke;
}
.jp-Notebook .jp-Cell {
    padding: 0 !important;
}
.jupyter-widgets.widget-tab {
    margin: 40px 30px;
}
.p-Collapse-header,
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current:before,
.jupyter-button ,
.widget-button,
.jupyter-widgets.widget-tab > .widget-tab-contents {
    border-radius: 4px;
    border: 1px solid #dcdcdc;
}
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab{
border-top-right-radius:4px;
border-top-left-radius: 4px;
border: 1px solid #dcdcdc;
}
.p-Collapse-contents{
border-bottom-right-radius:4px;
border-bottom-left-radius: 4px;
border: 1px solid #dcdcdc;
border-top:none;
}
.p-Collapse-open > .p-Collapse-header {
border-bottom-right-radius:0px;
border-bottom-left-radius: 0px;
}
body.jp-Notebook {
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    background: whitesmoke;
}
.widget-inline-hbox {
    margin: 0px;
}
.widget-output {
    height: auto !important;
    width: auto !important;
}
.jp-OutputArea {
    overflow-y: inherit;
}
.widget-text input[type="text"],
.widget-text input[type="number"],
.widget-dropdown > select {
   border-radius: 4px;
   box-shadow: inset 0 1px 1px rgb(0 0 0 / 5%);
}
.widget-vbox {
    justify-content: center;
    width: 100%;
    width: -webkit-fill-available !important;
    display: flex;
    border-radius: 4px;
    background: white;
    height: fit-content;
    padding: 20px;
   # box-shadow: 0px 1px 3px rgb(0 0 0 / 12%), 0px 1px 2px rgb(0 0 0 / 24%);
}
[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-light-background {
    width: 100%;
}
.jupyter-widgets.widget-tab {
    min-width: 30% !important;
}
html {
    background: #F2F2F2;
    height: 100%;
    margin: 0px;
    padding: 0px;
}
.widget-dropdown,
.widget-text{
    width: auto;
}
.experiment-label,
.widget-label {
    display: flex;
    align-items: center;
}
.experiment-label {
    justify-content: center;
}
.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
    background: white;
    padding-top: var(--jp-code-padding);
}
.jupyter-widgets.widget-tab > .widget-tab-contents {
    border-radius: 4px;
    border-top-left-radius: 0px;
    border-top-right-radius: 0px;
}
.widget-box {
    overflow: hidden !important;
}
.jp-OutputArea-output {
    overflow: hidden
    }
.jp-RenderedHTMLCommon p {
    margin-bottom: 0;
}
div.output_stderr {
    background: #fdd;
    display: none;
}
.main_container,
.connect_container_color {
background-color:  #F5F5F5;
margin-top: 3px;
padding:0;
}
.param_label{
font-weight: 800;
}
.config_label {
font-weight: 900;
font-size: 15px;
display:flex;
justify-content: center;
}
.jp-RenderedImage img {
    max-width: 100%;
    height: auto;
    width: 750px;
    margin-righ: 20px;
}
.plot-container {
    min-width: 750px;
}
.download-link{
    color: #192bc3;
}
.download-link:hover{
    color: blue;
}
div.port_ip_output pre{
    display: flex;
    justify-content: flex-end;
    border: none;
    align-content: stretch;
    align-items: flex-end;
    padding: 0;
}
div.output_area pre {
    flex-direction: column;
    word-break: break-word;
}
.plot-out{
    word-break: break-word;
    margin-right: 2%;
}
pre {
    border: none;
}
.widget-label { min-width: 10ex !important; }
.plot_inputs {
    width: auto !important;
    margin-top: 7.7%;
}
.jp-RenderedImage img {
    width: 100%;
}
.dash-reset-btn {
    diplay:flex;
    align-self: flex-end;
}
</style>"""
    )
)


header = HTML(
    f"""
<div class="nav" style="
margin: 0px;
background-color: #F2F2F2;
font-weight: bold;
box-shadow: 0px 1px 3px rgb(0 0 0 / 12%), 0px 1px 2px rgb(0 0 0 / 24%);
">
    <h1 style="
    text-align: center;
    font-family: helvetica;
    font-weight: 400;
    font-size: 20px;
    display: flex;
    align-items: center;
    padding:15px 20px;
    margin: 0px;
    hieght: 100px;
    jusify-content: flex-start;
    color:rgb(80,103,132)">
    <img src="./logo.png" alt="logo" width="50" height="50" style="margin-right: 10px;" >
    AEPsych Visualizer
    </h1>
</div>
"""
)

logs_style = Layout(width="600px", height="400px")
input_label_style = {"description_width": "initial"}
input_layout = Layout(margin="5px 0")
btn_style = Layout(margin="10px", width="95%")

btn_box_layout = Layout(display="flex", justify_content="flex-end")
file_output = widgets.Output(
    layout={"overflow": "scroll", "height": "50vh", "width": "100%", "padding": "5px"}
)

# ---------- Display Plot Inputs -----------
def on_value_change_input(change):
    inputs_output.clear_output()
    with inputs_output:
        change["new"]


yes_label = Text(
    value="Yes Trial",
    placeholder="yes_label",
    description="yes_label: ",
    disabled=False,
    style=input_label_style,
    layout=input_layout,
)
no_label = Text(
    value="No Trial",
    placeholder="no_label",
    description="no_label: ",
    disabled=False,
    style=input_label_style,
    layout=input_layout,
)
target_level = widgets.BoundedFloatText(
    value=0.75,
    min=0,
    max=1,
    step=0.1,
    description="target_level:",
    disabled=False,
    style=input_label_style,
    layout=input_layout,
)
cred_level = widgets.BoundedFloatText(
    value=0.95,
    min=0,
    max=1,
    step=0.1,
    description="cred_level:",
    disabled=False,
    style=input_label_style,
    layout=input_layout,
)
xlabel = Text(
    value="Angle (degrees)",
    placeholder="x axis label",
    description="xlabel: ",
    disabled=False,
    style=input_label_style,
    layout=input_layout,
)
ylabel = Text(
    value="Detection Probability",
    placeholder="y axis label",
    description="ylabel: ",
    disabled=False,
    style=input_label_style,
    layout=input_layout,
)
inputs_output = widgets.Output(
    layout={
        "border": "1px solid black",
        "padding": "20px",
        "margin": "20px",
        "min-height ": "100px",
    }
)
yes_label.observe(on_value_change_input, names="value")
no_label.observe(on_value_change_input, names="value")
target_level.observe(on_value_change_input, names="value")
cred_level.observe(on_value_change_input, names="value")
xlabel.observe(on_value_change_input, names="value")
ylabel.observe(on_value_change_input, names="value")
# -------- Select IP and Port ----------
server_ip = widgets.Text(
    placeholder="0.0.0.0",  # must use regex to check pattern and type int
    description="Server IP",
    disabled=False,
    layout=Layout(display="flex", flex_flow="column", margin="10px"),
)
port = widgets.Text(
    placeholder="5555",  # must add regex, type check, and error handling
    description="Port",
    disabled=False,
    layout=Layout(
        display="flex", flex_flow="column", margin="10px", justify_content="center"
    ),
)

connected_ip = widgets.Output()
connected_ip.add_class("port_ip_output")


def on_change_server_input(change):
    with inputs_output:
        change["new"]


server_ip.observe(on_change_server_input, names="value")
port.observe(on_change_server_input, names="value")

button_submit_port_ip = widgets.Button(
    description="Submit",
    disabled=False,
    button_style="info",
    tooltip="submit",
    layout=Layout(align_self="flex-end", width="fit-content", margin="10px"),
)


def submit_port_ip_clicked(b):
    """
    Configures the server to run on a port and ip address specified by the user.
    """
    global ip_address
    global server
    global client
    connected_ip.clear_output()
    try:
        with connected_ip:
            if len(port.value) and len(server_ip.value) == 0:
                print("Enter an ip address.")
                return

            if len(port.value) or len(server_ip.value):
                server = AEPsychServer(port.value, server_ip.value)
                connected_ip.clear_output()
                client = AEPsychClient(
                    ip=server_ip.value, port=port.value, server=server
                )
                make_config()

                ip_address = server_ip.value
                print(f"Connected by Client at {server_ip.value}   ")

            else:
                print(
                    "Server is connected to default port and IP address. Enter a port or ip address."
                )
    except Exception as e:
        print(f"Unable to connect at ip: {server_ip.value} and port {port.value}")
        print(e)
        print(traceback.format_exc())


button_submit_port_ip.on_click(submit_port_ip_clicked)
connection_input_box = widgets.Box(
    [server_ip, port, button_submit_port_ip],
    layout=Layout(justify_content="flex-end", margin="4px"),
)
connection_inputs = widgets.VBox(
    [connection_input_box, connected_ip],
    layout=Layout(min_height="15px", display="flex", justify_content="flex-end"),
)
connection_inputs.add_class("connect_container_color")
# -------- Config --------
config_output = widgets.Output()

button_submit_config = widgets.Button(
    description="Submit Config File",
    disabled=False,
    button_style="info",
    tooltip="submit",
)


def add_param(b):
    """
    Adds a new parameter field to the configuration generator form
    when the `Add Parameter` button is clicked.
    """
    global params_boxes
    global dim
    dim += 1
    name = widgets.Text(
        f"par{dim}",
        description="Name",
        layout=Layout(width="fit-content", margin="0px 2px"),
        style=input_label_style,
    )
    lb_input = widgets.FloatText(
        0.0,
        description=" Lower Bound:",
        layout=Layout(width="fit-content", margin="0px 2px"),
        style=input_label_style,
    )
    ub_input = widgets.FloatText(
        1.0,
        description=" Upper Bound:",
        layout=Layout(width="fit-content"),
        style=input_label_style,
    )
    hb = widgets.HBox(
        [name, lb_input, ub_input],
        layout=Layout(
            margin="2px 1px",
            display="flex",
            justify_content="flex-start",
            width="fit-content",
        ),
    )
    params_boxes.children = tuple(list(params_boxes.children) + [hb])
    pars = [child.children[0].value for child in params_boxes.children]
    lbs = [child.children[1].value for child in params_boxes.children]
    ubs = [child.children[2].value for child in params_boxes.children]


def remove_param(b):
    """
    Removes the last parameter field from the configuration generator form
    when the `Remove Parameter` button is clicked.
    """
    global params_boxes
    global dim
    if dim > 1:
        dim -= 1
        params_boxes.children = tuple(list(params_boxes.children[:-1]))


config_label = widgets.Label(value="Config Generator (Script)")
config_label.add_class("config_label")

params_label = widgets.Label(value="Parameters:")
params_label.add_class("param_label")
params_boxes = widgets.VBox([], layout=Layout(padding="2px"))
add_param(None)

add_param_btn = widgets.Button(description="Add Parameter", style=input_label_style)
add_param_btn.on_click(add_param)

rem_param_btn = widgets.Button(description="Remove Parameter", style=input_label_style)
rem_param_btn.on_click(remove_param)

params_btns = widgets.HBox(
    [add_param_btn, rem_param_btn], layout=Layout(margin="24px 0 4px 0")
)

threshold_input = widgets.BoundedFloatText(
    value=0.75,
    min=0,
    max=1.0,
    step=0.05,
    description="Target Threshold",
    style=input_label_style,
    layout=Layout(width="fit-content", margin="4px 0"),
)

stimuli_per_trial = widgets.Dropdown(
    options=[("Single", 0), ("Pairwise", 1)],
    value=0,
    description="Stimulus Type ",
    style=input_label_style,
    layout=Layout(width="fit-content", margin="4px 0"),
)

outcome_types = widgets.Dropdown(
    options=[("Binary", 0), ("Continuous", 1)],
    value=0,
    description="Response Type",
    style=input_label_style,
    layout=Layout(width="fit-content", margin="4px 0"),
)

experiment_method = widgets.Dropdown(
    options=[("Threshold", 0), ("Optimization", 1), ("Exploration", 2)],
    value=0,
    description="Experiment Type",
    style=input_label_style,
    layout=Layout(width="fit-content", margin="4px 0"),
)

initialize_model_on_start = widgets.Checkbox(
    value=False,
    description="Initialize Model on Start",
    style=input_label_style,
    disabled=False,
    indent=False,
)

init_min_total_tells = widgets.IntText(
    value=5,
    description="Number of Initialization trials ",
    style=input_label_style,
    disabled=False,
    layout=Layout(width="fit-content", margin="4px 0"),
)

opt_min_total_tells = widgets.IntText(
    value=50,
    description="Number of Optimization trials ",
    style=input_label_style,
    disabled=False,
    layout=Layout(width="fit-content", margin="4px 0"),
)

refit_every = widgets.IntText(
    value=5,
    description="Refit every: ",
    style=input_label_style,
    disabled=False,
    layout=Layout(width="fit-content", margin="4px 0"),
)

current_config_label = widgets.Label(value="Current Configuration")
current_config_label.add_class("param_label")
curr_config_label = widgets.Label(value="")


def reset_config_genorator_form():
    """
    Resets the config generator form values to the default values.
    """
    global config_output
    global init_min_total_tells
    global opt_min_total_tells
    global refit_every
    global curr_config_label
    global stimuli_per_trial
    global outcome_types
    global experiment_method
    global threshold_input
    global initialize_model_on_start
    global params_boxes

    config_output.clear_output()
    init_min_total_tells.value = 5
    opt_min_total_tells.value = 50
    refit_every.value = 5
    curr_config_label.value = ""
    stimuli_per_trial.value = 0
    outcome_types.value = 0
    experiment_method.value = 0
    threshold_input.value = 0.75
    initialize_model_on_start.value = False
    params_boxes.children = []


config_container = widgets.VBox(
    [
        config_label,
        config_output,
        init_min_total_tells,
        opt_min_total_tells,
        refit_every,
        current_config_label,
        curr_config_label,
        stimuli_per_trial,
        outcome_types,
        experiment_method,
        threshold_input,
        initialize_model_on_start,
        params_btns,
        params_label,
        params_boxes,
    ],
    layout=Layout(padding="8px", margin="0 2px 10px 2px"),
)

valid_combo = {
    "000": "SingleBinaryThreshold",
    "001": "SingleBinaryOptimization",
    "010": "SingleContinuousThreshold",
    "011": "SingleContinuousOptimization",
    "101": "PairwiseBinaryOptimization",
    "102": "PairwiseBinaryExploration",
}


def find_combo_name(combo_str):
    global valid_combo
    """
        takes in a string that represents the stimulus type, response type, and experiment type as an
        index that maps to the input parameter dropdowns.

        Identifies the valid combination and returns a dictionary with the valid configuration
        that should be set in the configuratation file.
    """
    valid_match = valid_combo[combo_str]
    curr_config_label.value = re.sub(r"(\w)([A-Z])", r"\1 \2", valid_match)

    if valid_match == "SingleBinaryThreshold":
        return {
            "init_generator": "SobolGenerator",
            "opt_generator": "OptimizeAcqfGenerator",
            "outcome_type": "[binary]",
            "acqf": "MCLevelSetEstimation",
            "model": "GPClassificationModel",
            "mean_covar_factory": "default_mean_covar_factory",
            "stimuli": 1,
            "target": threshold_input.value,
            "specifyAcqf": True,
            "objective": "ProbitObjective",
            "beta": 3.98,
        }
    elif valid_match == "SingleBinaryOptimization":
        return {
            "init_generator": "SobolGenerator",
            "opt_generator": "OptimizeAcqfGenerator",
            "outcome_type": "[binary]",
            "acqf": "qNoisyExpectedImprovement",
            "model": "GPClassificationModel",
            "mean_covar_factory": "default_mean_covar_factory",
            "stimuli": 1,
            "target": -1,
            "specifyAcqf": True,
            "objective": "ProbitObjective",
        }
    elif valid_match == "SingleContinuousThreshold":
        return {
            "init_generator": "SobolGenerator",
            "opt_generator": "OptimizeAcqfGenerator",
            "outcome_type": "[continuous]",
            "acqf": "MCLevelSetEstimation",
            "model": "GPRegressionModel",
            "mean_covar_factory": "",
            "stimuli": 1,
            "target": threshold_input.value,
            "specifyAcqf": True,
            "objective": "IdentityMCObjective",
        }
    elif valid_match == "SingleContinuousOptimization":
        return {
            "init_generator": "SobolGenerator",
            "opt_generator": "OptimizeAcqfGenerator",
            "outcome_type": "[continuous]",
            "acqf": "qNoisyExpectedImprovement",
            "model": "GPRegressionModel",
            "mean_covar_factory": "",
            "stimuli": 1,
            "target": -1,
            "specifyAcqf": False,
            "objective": "",
        }
    elif valid_match == "PairwiseBinaryOptimization":
        return {
            "init_generator": "PairwiseSobolGenerator",
            "opt_generator": "PairwiseOptimizeAcqfGenerator",
            "outcome_type": "[binary]",
            "acqf": "qNoisyExpectedImprovement",
            "model": "PairwiseProbitModel",
            "mean_covar_factory": "default_mean_covar_factory",
            "stimuli": 2,
            "target": -1,
            "specifyAcqf": True,
            "objective": "ProbitObjective",
        }
    elif valid_match == "PairwiseBinaryExploration":
        return {
            "init_generator": "PairwiseSobolGenerator",
            "opt_generator": "PairwiseOptimizeAcqfGenerator",
            "outcome_type": "[binary]",
            "acqf": "PairwiseMCPosteriorVariance",
            "model": "PairwiseProbitModel",
            "mean_covar_factory": "default_mean_covar_factory",
            "stimuli": 2,
            "target": -1,
            "specifyAcqf": True,
            "objective": "ProbitObjective",
        }


def string_builder(combo_res):
    """
    Builds the configuration string based on the inputs from the config
    generator and returns the string.
    """

    global outcome

    init_min_tells = init_min_total_tells.value
    start_with_model = initialize_model_on_start.value
    opt_min_tells = opt_min_total_tells.value

    pars = [child.children[0].value for child in params_boxes.children]
    parnames = f"[{','.join(par for par in pars)}]"
    lbs = [child.children[1].value for child in params_boxes.children]
    ubs = [child.children[2].value for child in params_boxes.children]
    refit = refit_every.value

    acqf = combo_res["acqf"]
    model = combo_res["model"]
    init_generator_val = combo_res["init_generator"]
    opt_generator_val = combo_res["opt_generator"]
    outcome_type = combo_res["outcome_type"]
    stimuli = combo_res["stimuli"]
    target = combo_res["target"]
    mean_covar_factory = combo_res["mean_covar_factory"]
    specify_acqf = combo_res["specifyAcqf"]
    objective = combo_res["objective"]

    if outcome_types.value == 1:
        outcome = widgets.BoundedFloatText(
            value=0,
            step=0.1,
            description="outcome",
            disabled=False,
            style=input_label_style,
            layout=input_layout,
        )

    beta = ""
    if "beta" in combo_res:
        beta = combo_res["beta"]

    # Added to config conditionally
    tartget_section = f"target = {target}\n" if float(target) >= 0 else ""
    start_with_model_section = (
        f"model = {model} \n" if initialize_model_on_start.value else ""
    )
    mean_covar_factory_section = (
        f"mean_covar_factory = {mean_covar_factory}" if mean_covar_factory != "" else ""
    )

    # Scale some parameters based on number of dimension
    samps = 500 * (len(pars) + 1)
    inducing_size = 50 * len(pars)
    restart = 10

    sb = f"""
    # Common Attributes
    [common]
    parnames = {parnames} # names of the parameters
    lb = {lbs} # lower bound of the parameter
    ub = {ubs} # upper bound of parameter
    stimuli_per_trial = {stimuli} # the number of stimuli shown in each trial; 1 for single, or 2 for pairwise experiments
    outcome_types = {outcome_type} # the type of response given by the participant; can be [binary] or [continuous]
    {tartget_section}
    strategy_names = [init_strat, opt_strat] # The strategies that will be used, corresponding to the named sections below

    # Initial Strategy
    [init_strat]
    min_tells = {init_min_tells}
    generator = {init_generator_val}
    {start_with_model_section}

    # Optimized Strategy
    [opt_strat]
    min_tells = {opt_min_tells}
    refit_every = {refit}
    generator = {opt_generator_val}
    acqf = {acqf}
    model = {model}

    # Model Parameters
    [{model}]
    inducing_size = {inducing_size}
    {mean_covar_factory_section}

    # Optimized Generator Parameters
    [{opt_generator_val}]
    restarts = {restart}
    samps = {samps}

    """
    if specify_acqf == True:
        sb += f"[{acqf}] \n"
        if beta != "":
            sb += f"    beta = {beta} \n"
        if objective != "":
            sb += f"    objective = {objective} \n"

    return sb


def validate_combination(combo_str):
    """
    Validates the stimulus, response, and method input combination and
    checks it agains the `valid_combo` dictionary.

    If the combination is valid returns a `combo_string` (Ex: "SingleBinaryThreshold")
    with a description of the experiment type.  If the communication is not valid and
    empty string is  returned.
    """

    global valid_combo

    stimulus = combo_str[0]
    response = combo_str[1]
    method = combo_str[2]
    if combo_str in valid_combo:  # If already valid, do nothing
        return combo_str

    else:
        return ""


def make_config():
    """
    Configures the server by sending the config string to the server via
    the aepsych_client.
    """

    global dim
    global client

    dim = len(params_boxes.children)
    param_output.clear_output()
    combo_str = (
        f"{stimuli_per_trial.value}{outcome_types.value}{experiment_method.value}"
    )
    is_valid = validate_combination(combo_str)

    if is_valid == "":
        config_output.clear_output()
        with config_output:
            if stimuli_per_trial.value == 1 and outcome_types.value == 1:
                print("Pairwise only supports binary outcomes")
            else:
                print(
                    """
                Invalid configuration combo of stimulus, response, and experiment type.
                please refer to the config docs https://aepsych.org/docs/configs
                """
                )
            return

    valid_combo_res = find_combo_name(combo_str)
    config = string_builder(valid_combo_res)

    try:
        config_output.clear_output()

        with config_output:
            if client != None:
                config_status = None

                if config_status == None:
                    button_submit_config.disable = True
                    print("Sending config...")

                client.configure(config_str=config)
                db_strats.options = [
                    (f"strat - {i}", j) for i, j in enumerate(client.configs)
                ]
                db_strats.value = db_strats.options[-1][1]
                config_status = True
                button_submit_config.disable = False

                if config_status:
                    config_output.clear_output()

                get_next("set_inputs")
                if dim > 3:
                    display(
                        HTML(
                            """
                    <p style="text-align: center;">
                        Config sent to server successfully
                        *Please note: No plots for >3d!*
                    </p>"""
                        )
                    )
                else:
                    display(
                        HTML(
                            """
                    <p style="text-align: center;">
                        Config sent to server successfully
                    </p>"""
                        )
                    )

            else:
                print("Server not connected")
    except Exception as e:
        print("Unable to configure server.")
        print(e)
        print(traceback.format_exc())


def clicked_submit_config(b):
    make_config()


button_submit_config.on_click(clicked_submit_config)

config_buttons = widgets.HBox(
    [button_submit_config],
    layout=Layout(display="flex", justify_content="flex-end", marging="5px 0"),
)
# Resume Config
resume_config_output = widgets.Output(layout=Layout(margin="10px"))

resume_exp_output = widgets.Output(layout=Layout(margin="10px"))

current_db_label = widgets.Label(value="None")
db_experiments = widgets.Dropdown(
    options=[("-No Experiments-", 0)],
    value=0,
    style=input_label_style,
    description="Select Experiment:",
    layout=Layout(width="fit-content"),
    disabled=False,
)

select_exp_btn = widgets.Button(
    description="Submit",
    disabled=False,
    button_style="info",
    tooltip="submit",
    layout=Layout(align_self="flex-end", width="fit-content", margin="10px"),
)
resume_exp_container = widgets.Box(
    [db_experiments, select_exp_btn],
    layout=Layout(
        display="flex",
        flex_flow="row",
        justify_content="flex-end",
        align_items="center",
        padding="35px",
    ),
)


def select_exp_clicked(change):
    """
    Replays data for an experiment based on the UUID selected via a dropdown.
    """

    global dim
    global client
    resume_exp_output.clear_output()
    with resume_exp_output:
        db_experiments.disable = True
        print(f"Replaying data for experiment: {db_experiments.value}")
        client.server.replay(db_experiments.value, skip_computations=True)
        current_db_label.value = f"Experiment ID: {db_experiments.value}"
        print("loading...")
        db_experiments.disable = False
        client.ask()
        get_next("set_inputs")
        dim = len(client.server.parnames)
        display_data()  # loads config on inital render
        resume_config_value()  # reloads page


select_exp_btn.on_click(select_exp_clicked)


def set_db_exp_record_values():
    """
    Loads the experiments in the database as options for the experiment dropdown.
    """

    exp_ids_test = client.server.db.get_master_records()
    exp_opt_list = []
    i = 0
    for rec in exp_ids_test:
        i += 1
        exp_opt_list.append((f"exp{i} - {rec.experiment_name}", rec.experiment_id))
    db_experiments.options = exp_opt_list

    return exp_opt_list


def resume_config_value():
    """
    Displays the config string for the current strategy when an experiment is resumed.
    """

    global client
    resume_config_output.clear_output()
    resume_exp_output.clear_output()
    with resume_config_output:
        display(current_db_label)
        display(
            html(
                f"""
        <div style='height: 400px;width:100%'>
            <pre
            style='font-family: monospace;background-color: black;
            color: green;padding: 8px;height: 100%;overflow: auto;'
            >{client.server.config}</pre>

        </div>
        """
            )
        )
        try:
            set_db_exp_record_values()
        except Exception as e:
            print(e)
            print(traceback.format_exc())


# --------- Accordion -----------
button_submit = widgets.Button(
    description="Submit",
    disabled=False,http://localhost:8888/notebooks/visualizer/AEPsych_Visualizer_Dash_Beta.ipynb#
    button_style="info",
    tooltip="submit",
    layout=btn_style,
)
accordion = widgets.Accordion(
    children=[
        widgets.VBox([no_label, yes_label], layout=Layout(padding="5px")),
        widgets.VBox(
            [
                target_level,
                cred_level,
            ],
            layout=Layout(padding="5px"),
        ),
        widgets.VBox([xlabel, ylabel], layout=Layout(padding="5px")),
    ],
    layout=Layout(min_width="280px"),
)
accordion.set_title(0, "Outcome Labels")
accordion.set_title(1, "Parameters")
accordion.set_title(2, "Axis Labels")


def submit():
    file_output.clear_output()
    with file_output:
        display_plot()


def submit_clicked(b):
    submit()


button_submit.on_click(submit_clicked)
plot_interact_box = widgets.VBox([accordion, button_submit])
plot_interact_box.add_class("plot_inputs")
# -------- Log View--------
logger_output = widgets.Output()


def load_logs():
    """Loads server logs."""
    logger_output.clear_output()
    log_str = ""

    log_path = os.path.join(os.getcwd(), "logs", "bayes_opt_server.log")

    logger_output.clear_output()
    with logger_output:
        if os.path.exists(log_path):
            with open(log_path) as f:
                contents = f.readlines()
                for log in contents:
                    log_str += (
                        f"<p style='margin-top: 1px;margin-bottem: 1px'>{log}</p>"
                    )
            display(
                html(
                    f"""
            <div style='height: 500px;'>
                <pre
                style='font-family: monospace; background-color: black; color: white;
                padding: 4px; height: 100%;'
                >{log_str}</pre>
            </div>
            """
                )
            )
        else:
            print("Log directory was not generated")


# --------- Interactive View------------
param_output = widgets.Output(
    layout=Layout(
        padding="2px",
        height="auto",
        align_items="center",
        justify_conten="center",
        min_width="400px",
        width="auto",
    )
)

outcome = widgets.Dropdown(
    options=[("0", 0), ("1", 1)],
    value=0,
    description="outcome",
    style=input_label_style,
    layout=Layout(margin="5px 10px", diplay="flex", justify_content="space-between"),
)

param_box = widgets.VBox([], layout=Layout(max_width="350px"))

interactive_btn_labels = ["New Parameters", "Update Model", "Send to Client"]

interactive_buttons = [
    widgets.Button(
        description=f"{interactive_btn_labels[i]}",
        disabled=False,
        button_style="",
        tooltip="Click me",
    )
    for i in range(3)
]

inteact_btn_box = widgets.Box(interactive_buttons)
interactive_buttons[2].disabled = True
interactive_buttons[1].disabled = True
interactive_container = widgets.VBox([inteact_btn_box, param_box, param_output])


def pairwise_check(config_str):
    """Checks config string to determin if it is a pairwise
    experiment and returns true of false:
    """
    pairwise_substring = "stimuli_per_trial = 2"
    if search(pairwise_substring, str(config_str)):
        return True
    else:
        return False



import json

def gen_param_bounds():
    """Generates a dictionary with the param name and the corresponding lb and ub"""
    global client

    param_bounds = {}
    config_common = dict(client.server.config["common"])
    param_names = config_common["parnames"].strip('][').split(', ')
    lb = config_common["lb"].strip('][').split(', ')
    ub = config_common["ub"].strip('][').split(', ')

    for i in range(len(param_names)):
        param_bounds[param_names[i]] = {"min_val": float(lb[i]), "max_val": float(ub[i])}

    return param_bounds






def get_next(status):
    """
    Sends an ask message to the client and updates the values to be displayed in the param inputs.
    """

    global client

    interactive_buttons[1].disabled = False

    if status == "set_inputs":
        param_output.clear_output()
        with param_output:
            print(client.server.parnames)

            param_bounds = gen_param_bounds()


            if pairwise_check(client.server.config):
                children = []
                for name in client.server.parnames:
                    children.append(
                        widgets.FloatText(
                            value=0,
                            description=f"{name}",
                            disabled=False,
                            layout=Layout(margin="5px 10px"),
                        )
                    )
                    children.append(
                        widgets.FloatText(
                            value=0,
                            description=f"{name}",
                            disabled=False,
                            layout=Layout(margin="5px 10px"),
                        )
                    )

                outcome.value = 0
                children.append(outcome)
                param_box.children = children
                return
            else:
                children = []
                for name in client.server.parnames:
                    children.append(
                        widgets.BoundedFloatText(
                            value=0,
                            description=f"{name}",
#                             min= param_bounds[name]["min_val"],
#                             max = param_bounds[name]["max_val"],
                            disabled=False,
                            layout=Layout(margin="5px 10px"),
                        )
                    )
                outcome.value = 0
                children.append(outcome)
                param_box.children = children
                return

    param_output.clear_output()

    ask_status = None
    if ask_status == None:
        with param_output:
            interactive_buttons[0].disabled = True
            interactive_buttons[1].disabled = True
            print("Loading... model is generating new points")
    res = client.ask()
    ask_response = res["config"]
    ask_status = True
    interactive_buttons[0].disabled = False
    interactive_buttons[1].disabled = False

    if ask_status:
        param_output.clear_output()
        with param_output:
            print("Received msg [ask]")
            if res["is_finished"]:
                print(f"All strategies in the experiment have finished.")

            if pairwise_check(client.server.config):
                children = []
                for name, data in ask_response.items():
                    children.append(
                        widgets.FloatText(
                            value=data[0],
                            description=f"{name}",
                            disabled=False,
                            layout=Layout(
                                margin="5px 10px", width="fit-content !important"
                            ),
                        )
                    )
                    children.append(
                        widgets.FloatText(
                            value=data[1],
                            description=f"{name}",
                            disabled=False,
                            layout=Layout(
                                margin="5px 10px", width="fit-content !important"
                            ),
                        )
                    )
            else:
                children = []
                for name, data in ask_response.items():
                    children.append(
                        widgets.FloatText(
                            value=data[0],
                            description=f"{name}",
                            disabled=False,
                            layout=Layout(
                                margin="5px 10px", width="fit-content !important"
                            ),
                        )
                    )

            outcome.value = 0
            children.append(outcome)
            param_box.children = children


def tell_model(d):
    """
    Sends an ask message to the client based on the values in the param inputs.

    Renders two inputs for each parameter when running pairwise experiments.
    """

    global client

    param_output.clear_output()
    tell_msg = {}

    if pairwise_check(client.server.config):
        for param in param_box.children:
            if param.description != "Outcome: ":
                if param.description in tell_msg:
                    tell_msg[param.description].append(param.value)
                else:
                    tell_msg[param.description] = [param.value]
    else:
        for param in param_box.children:
            if param.description != "Outcome: ":
                tell_msg[param.description] = param.value

    outcome_val = outcome.value

    try:
        client_status = None

        if client_status == None:
            with param_output:
                interactive_buttons[0].disabled = True
                interactive_buttons[1].disabled = True
                print("Sending tell message...")

        client.tell(config=tell_msg, outcome=outcome_val)

        client_status = True
        interactive_buttons[0].disabled = False
        interactive_buttons[1].disabled = False

        if client_status:
            param_output.clear_output()
            with param_output:
                print("Received msg [tell]")
            for child in param_box.children:
                child.value = 0
    except Exception as e:
        with param_output:
            print("Unable to update Model...")
            print(e)
            print(traceback.format_exc())


def send_to_client(d):
    """Will send a generic message of any type to the client"""

    param_output.clear_output()
    with param_output:
        print("""Send to Client - Not yet implemented""")


interactive_buttons[0].on_click(get_next)
interactive_buttons[1].on_click(tell_model)
interactive_buttons[2].on_click(send_to_client)

# -------- Table view ----------
table_data_output = widgets.Output(layout=Layout(width="600px", height="400px"))
download_link_output = widgets.Output()


def create_download_html(htmlWidget, filename, title="Click here to download: "):
    """Creates a download link for .csv and .db files"""

    htmlWidget.value = '<i class="fa fa-spinner fa-spin fa-2x fa-fw"></i><span class="sr-only">Loading...</span>'

    data = open(filename, "rb").read()
    b64 = base64.b64encode(data)
    payload = b64.decode()

    html = '<a download="{filename}" class="download-link" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    htmlWidget.value = html.format(
        payload=payload, title=title + filename, filename=filename
    )


htmlWidget = widgets.HTML(value="")
csvWidget = widgets.HTML(value="")


def download_link(csv_data):
    """Generates the path for the download link.
    Writes the experiment data to a .csv file.
    Passes the path, filename, and link label to create_download_html() to generate a
    download link for .db and .csv files.
    """
    global ip_address
    global resume_experiment
    global new_expriment

    download_link_output.clear_output()
    with download_link_output:
        csv_file_name = os.path.join(".", "databases", "aepsych_data.csv")
        db_file_name = os.path.join(".", "databases", "default.db")

        csv_path = os.getcwd() + csv_file_name[1:]

        f = open(csv_path, "w")
        csv_data.to_csv(csv_path, index=False)
        create_download_html(csvWidget, csv_file_name, "Download .csv: ")
        display(csvWidget)

        if new_experiment and ip_address != "":

            f_content = open(os.path.join(f"{ip_address}"), "rb")
            file_name = f"{f_content.name}.db"

            ip_path = os.path.join(".", "databases", file_name)
            new_f = open(ip_path, "wb")

            with open(ip_path, "wb") as f:
                new_f.write(f_content.read())
                f.close()

            create_download_html(htmlWidget, ip_path, "Download .db: ")
            display(htmlWidget)

        elif new_experiment:
            create_download_html(htmlWidget, db_file_name, "Download .db: ")
            display(htmlWidget)

        elif resume_experiment:
            file_name = None
            for name, file_info in uploader.value.items():
                file_name = name

            relative_path = os.path.join(".", "databases", file_name)
            create_download_html(htmlWidget, relative_path, "Download .db: ")
            display(htmlWidget)


def display_data():
    """Creates a DataFrame with experiment data."""

    global client
    table_data_output.clear_output()

    with table_data_output:
        try:
            if client.server.strat.x is not None:
                if pairwise_check(client.server.config):
                    data = {}
                    for i, par in enumerate(client.server.parnames):
                        for par_values in client.server.strat.x[:, i]:
                            val_zero = par_values[0]
                            val_zero = f"{val_zero}".replace(")", "")
                            val_zero = f"{val_zero}".replace("tensor(", "")

                            val_one = par_values[1]
                            val_one = f"{val_one}".replace(")", "")
                            val_one = f"{val_one}".replace("tensor(", "")

                            data[par] = val_zero
                            data[f"{par} "] = val_one

                else:
                    data = {
                        par: client.server.strat.x[:, i]
                        for i, par in enumerate(client.server.parnames)
                    }

                data["outcome"] = client.server.strat.y
                data = pd.DataFrame(data)
                csv_data = data
                download_link(csv_data)
                display(
                    html(
                        f"""<div style='height:400px;overflow:auto;width:100%; padding:5px; '>
                {data.style.render()}
                </div>"""
                    )
                )
            else:
                print("Collect more data to display a Data Frame")

        except IndexError as err:
            print("Collect more data")
        except Exception as e:
            print(e)
            print(traceback.format_exc())


# -------- Plot view ----------
plot_output = widgets.Output(
    layout=Layout(
        padding="2px",
        height="100%",
        align_items="center",
        justify_conten="center",
        width="100%",
    )
)
plot_output.add_class("plot-out")

current_strat_label = widgets.Label(value="None")
db_strats = widgets.Dropdown(
    options=[("-No Strats-", 0)],
    value=0,
    style=input_label_style,
    description="Select Strat:",
    layout=Layout(width="fit-content"),
    disabled=False,
)

resume_strat = widgets.Button(
    description="Submit",
    disabled=False,
    button_style="info",
    tooltip="submit",
    layout=Layout(align_self="flex-end", width="fit-content", margin="10px"),
)


def resume_strat_btn(change):
    """
    Resumes a strategy based on the strategy selected in the db_strat dropdown.
    """

    global dim
    global client

    client.resume(config_id=db_strats.value)
    client.ask()
    current_strat_label.value = (
        f"Current Strat: {db_strats.options[db_strats.value][0]}"
    )
    dim = len(client.server.parnames)
    display_plot()
    get_next("set_inputs")


resume_strat.on_click(resume_strat_btn)


def display_plot():
    """
    Displays a model of the current strategy.

    Supports 1 to 3 dimensional plotting.
    """

    global strat
    plot_output.clear_output()
    with plot_output:

        error = False
        with plot_output:  # Error Handling
            plot_output.clear_output()
            if len(ylabel.value) > 35:
                error = True
                print("ylabel must be less than 35 characters ")
            elif len(xlabel.value) > 35:
                error = True
                print("xlabel must be less than 35 characters ")
            elif len(yes_label.value) > 35:
                error = True
                print("yes_label must be less than 35 characters ")
            elif len(no_label.value) > 35:
                error = True
                print("no_label must be less than 35 characters ")

        if error == False:
            inputs_output.clear_output()
        try:
            client.load_config_index()
            db_strats.options = [
                (f"strat - {i}", j) for i, j in enumerate(client.configs)
            ]

            current_strat_label.value = (
                f"Current Strat: {db_strats.options[db_strats.value][0]}"
            )
            display(current_strat_label)

            if dim == 1:
                xlabel.value = client.server.parnames[0]
                ylabel.value = "Response Probability"
            if dim == 2:
                xlabel.value = client.server.parnames[0]
                ylabel.value = client.server.parnames[1]

            if client.server.strat == None:
                print("Server has not strat.")
                return
            elif client.server.strat.model is not None and dim == 3:
                plot_interact_box.layout.display = "none"
                plot_strat_3d(client.server.strat, parnames=client.server.parnames)
            elif client.server.strat.model is not None and dim < 4:
                plot_interact_box.layout.display = "inline"
                plot_strat(
                    client.server.strat,
                    xlabel=xlabel.value,
                    ylabel=ylabel.value,
                    no_label=no_label.value,
                    yes_label=yes_label.value,
                    cred_level=cred_level.value,
                    target_level=target_level.value,
                )
            elif dim > 3:
                print("Please note: No plots for >3d!")
            elif client.server.strat.model == None:
                print("Collect more data to build the model and create the plot")

            display(
                widgets.HBox(
                    [db_strats, resume_strat],
                    layout=Layout(
                        display="flex",
                        justify_content="flex-start",
                        align_items="center",
                    ),
                )
            )

        except AttributeError as err:
            print("Collect more data to build the model and create the plot, \n")
            print(err)
            print(traceback.format_exc())
        except NotImplementedError as err:
            print(err)
            print("\n Only plot 3 pramameters at once")
        except AssertionError as err:
            print("Collect more data to build the model and create the plot, \n")
            print(err)
            print(traceback.format_exc())
        except Exception as e:
            print(e)
            print(traceback.format_exc())


out = widgets.Output(
    layout=Layout(
        padding="2px",
        height="100%",
        align_items="center",
        justify_conten="flex-start",
        margin="0",
        width="100%",
    )
)


exp_details = widgets.Button(
    description="Exp Details", disabled=False, button_style="", tooltip="Click me"
)
plot_view = widgets.Button(
    description="Plot View", disable=False, button_style="", tooltip="Click me"
)
table_view = widgets.Button(
    description="Table View", disable=False, button_style="", tooltip="Click me"
)
logs = widgets.Button(
    description="Logs", disable=False, button_style="", tooltip="Click me"
)
interactive = widgets.Button(
    description="Interactive", disable=False, button_style="", tooltip="Click me"
)

navigation = widgets.Box(
    [exp_details, plot_view, table_view, logs, interactive],
    layout=Layout(
        display="flex",
        flex_flow="column",
        width="200px",
        margin="0 2.5%",
        min_width="152px",
    ),
)
output_header_style = Layout(justify_content="center")
nav_display_style = Layout(display="flex", flex_flow="column")


container_one = widgets.Box(
    [
        Label("Experiment details", layout=output_header_style),
        resume_config_output,
        resume_exp_container,
        resume_exp_output,
    ],
    layout=nav_display_style,
)


container_one_new = widgets.Box(
    [config_container, config_buttons], layout=nav_display_style
)

plot_containter = widgets.HBox([plot_output, plot_interact_box])
plot_containter.add_class("plot-container")

container_two = widgets.Box(
    [Label("Plot view", layout=output_header_style), plot_containter],
    layout=Layout(width="100%", display="flex", flex_flow="column"),
)

container_three = widgets.Box(
    [
        Label("Table View", layout=output_header_style),
        download_link_output,
        table_data_output,
    ],
    layout=nav_display_style,
)

container_four = widgets.Box(
    [Label("Logs", layout=output_header_style), logger_output], layout=nav_display_style
)

container_five = widgets.Box(
    [interactive_container], layout=Layout(margin="0", padding="0")
)

nav_container = widgets.Box(
    [navigation, out], layout=widgets.Layout(display="flex", flex_flow="row")
)
main_container = widgets.Box(
    [connection_inputs, nav_container],
    layout=widgets.Layout(display="flex", flex_flow="column", marging="auto"),
)

main_container.add_class("main_container")


def nav_btn_click(b):
    """
    Manages the dashboard navigation and displays the selected view.
    """

    global resume_experiment
    global new_experiment

    show_display = b.description
    out.clear_output()
    if show_display == "Exp Details":
        with out:
            if new_experiment:
                display(container_one_new)
            if resume_experiment:
                resume_config_value()
                display(container_one)

    elif show_display == "Plot View":
        display_plot()
        with out:
            display(container_two)

    elif show_display == "Table View":
        display_data()
        with out:
            display(container_three)

    elif show_display == "Logs":
        with out:
            load_logs()
            display(container_four)

    elif show_display == "Interactive":

        with out:
            display(container_five)


exp_details.on_click(nav_btn_click)
plot_view.on_click(nav_btn_click)
table_view.on_click(nav_btn_click)
logs.on_click(nav_btn_click)
interactive.on_click(nav_btn_click)


# ------ View 1 ------
uploader = widgets.FileUpload(
    description="Connect to Database",
    accept=".db",
    multiple=False,
    layout=Layout(margin="15px 60px"),
)


def start_server():
    """
    Starts a server when the `Build Experiment` button is clicked.

    Passes the server to the aepsych_client and renders the dashboard home view.
    """

    global strat
    global server
    global client

    with error_output:
        try:
            server = AEPsychServer()
            client = AEPsychClient(server=server)
            make_config()
            get_next("set_inputs")

            config_output.clear_output()
            with out:
                display(container_one_new)

            home_view_container.layout.display = "none"
            main_container.layout.display = "inline"
            button_reset.layout.display = "inline"
            display(main_container)
        except Exception as e:
            print("Unable to start a new experiment")
            print(e)
            print(traceback.format_exc())


def resume_server():
    """
    Starts a server with a specified database_path when the `Connect to Database` button is clicked.

    The server is passed to the aepsych_client and the last experiment in the db is replayed.
    The last strategy in the experiment is resumed and the dashboard home view is rendered.
    """

    global server
    global client
    global dim
    global selected_exp_db_flag

    with error_output:
        try:
            server = AEPsychServer(database_path=database_path)
            client = AEPsychClient(server=server)
            exp_ids = [rec.experiment_id for rec in server.db.get_master_records()]

            print("Replaying experiment data...")
            client.server.replay(
                exp_ids[-1], skip_computations=True
            )  # replays last experiment
            current_db_label.value = f"Experiment ID: {exp_ids[-1]}"

            client.load_config_index()
            db_strats.options = [
                (f"strat - {i}", j) for i, j in enumerate(client.configs)
            ]
            db_strats.value = db_strats.options[-1][1]
            client.resume(config_id=db_strats.value)
            client.ask()

            get_next("set_inputs")
            dim = len(client.server.parnames)

            display_data()  # loads config on inital render
            resume_config_value()
            with out:
                display(container_one)
            home_view_container.layout.display = "none"
            main_container.layout.display = "inline"
            button_reset.layout.display = "inline"

        except IndexError as err:
            error_output.clear_output()
            print(
                """
            File does not contain master records with experiment id(s).
            Please upload another file.
            """
            )
            print(err)
            print(traceback.format_exc())
        except Exception as e:
            print("unable to connect to server...")
            print(e)
            print(traceback.format_exc())


def upload():
    """
    Opens the uploaded file and writes the content to a file path in the database directory.

    Creates a databases directory if one does not exist and updates the database_path name.
    """

    global strat
    global database_path
    global db_file_name
    error_output.clear_output()
    with error_output:
        error_output.clear_output()
        if uploader.value == {}:
            print("No file uploaded")
        else:
            try:
                for db_file_name, file in uploader.value.items():
                    db_file_name

                    filepath = os.path.join(os.getcwd(), "databases", db_file_name)

                    if not os.path.exists(os.path.join(os.getcwd(), "databases")):
                        os.makedirs(os.path.join(os.getcwd(), "databases"))

                    with open(filepath, "wb") as f:
                        f.write(file["content"])
                        if filepath != None:
                            database_path = filepath
                        resume_server()
                        f.close()
            except Exception as e:
                print("File upload unsuccessful.")
                print(e)
                print(traceback.format_exc())


def uploader_on_change(change):
    global resume_experiment

    error_output.clear_output()
    with error_output:
        try:
            upload()
            resume_experiment = True
        except Exception as e:
            print(e)
            print(traceback.format_exc())


uploader.observe(uploader_on_change, names="value")

error_output = widgets.Output(
    layout=Layout(
        padding="2px",
        height="100%",
        align_items="center",
        justify_conten="center",
        width="100%",
    )
)
build_experiment = widgets.Button(
    description="Build Experiment",
    disabled=False,
    button_style="info",
    tooltip="Click me",
    layout=Layout(width="auto", height="auto", margin="15px 60px"),
)

experiment_label = Label("Build a new experiment, or load data from an old one")
experiment_label.add_class("experiment-label")
home_view_container = widgets.Box(
    [
        experiment_label,
        widgets.HBox(
            [build_experiment, uploader],
            layout=widgets.Layout(
                display="flex",
                justify_content="center",
                margin="30px 0",
                align_items="center",
            ),
        ),
        error_output,
    ],
    layout=Layout(
        display="flex",
        flex_flow="column",
        width="100%",
        height="100%",
        margin="60px 0",
        justify_content="center",
        align_items="center",
    ),
)


wrapper_container = widgets.Box(
    [header, reset_button_container, main_container, home_view_container],
    layout=Layout(display="flex", flex_flow="column"),
)
main_container.layout.display = "none"
button_reset.layout.display = "none"
display(wrapper_container)


def build_experiment_clicked(b):
    global new_experiment
    new_experiment = True
    error_output.clear_output()
    with error_output:
        start_server()


build_experiment.on_click(build_experiment_clicked)


def reset_dash(e):
    """
    Resets the dashboard state and renders the home view.
    """
    global client
    global strat
    global db_file_name
    global database_path
    global new_experiment
    global resume_experiment
    global dim
    global ip_address
    global server

    with connect_out:
        server = None
        connect_out.clear_output()

    client = None
    strat = None
    db_file_name = None
    database_path = None
    new_experiment = False
    resume_experiment = False
    dim = 0
    ip_address = ""

    reset_config_genorator_form()
    add_param(None)
    clear_logs()
    clear_databases_dir()

    out.clear_output()
    file_output.clear_output()
    inputs_output.clear_output()
    connected_ip.clear_output()
    resume_config_output.clear_output()
    resume_exp_output.clear_output()
    logger_output.clear_output()
    param_output.clear_output()
    table_data_output.clear_output()
    download_link_output.clear_output()
    plot_output.clear_output()
    out.clear_output()
    error_output.clear_output()
    home_view_container.layout.display = "inline"
    main_container.layout.display = "none"
    button_reset.layout.display = "none"


button_reset.on_click(reset_dash)
