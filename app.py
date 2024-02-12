import base64
import io

import dash
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import shap
import xgboost as xgb
from dash import Dash, dcc, html, Input, Output, State, callback
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("loading training data")
# tat = pd.read_csv("./data/16_lucas_organic_carbon_training_and_test_data.csv")
# targets = pd.read_csv("./data/16_lucas_organic_carbon_target.csv")

tat = pd.read_csv("./data/lucas_organic_carbon_training_and_test_data_NEW.csv")
targets = pd.read_csv("./data/lucas_organic_carbon_target.csv")
tat_feature_names = tat.columns.to_list()
tat_train, tat_test, targets_train, targets_test = train_test_split(tat, targets, test_size=0.2, random_state=42)

print("loading models")
# NAME, Model, type, explanation
# model_list = [
#     (
#         'XGBoost',
#         joblib.load('./new_models/xgboost.pkl'),
#         'xgboost',
#         joblib.load('./shapley-values/saved_values/xgboost-shapley_values.json')
#     ),
#     (
#         'Random Forest',
#         joblib.load('./new_models/random_forest.pkl'),
#         'random_forest',
#         joblib.load('./shapley-values/saved_values/randomforest-shapley_values.json')
#     ),
#     (
#         'Logistic Regression',
#         joblib.load('./new_models/lr_clf.pkl'),
#         'logic_regression',
#         joblib.load('./shapley-values/saved_values/lr_clf-shapley_values.json')
#     )
# ]

model_list = [
    (
        'XGBoost',
        joblib.load('./models/best_model_xgboost.pkl'),
        'xgboost',
        joblib.load('./shapley-values/saved_values/xgboost-shapley_values_old')
    ),
    (
        'Logistic Regression',
        joblib.load('./models/logreg.pkl'),
        'logic_regression',
        joblib.load('./shapley-values/saved_values/logreg_shapley_values_old')
    )
]


model_predictions_df = pd.DataFrame([])
label_encoder = LabelEncoder()


def add_predictions_to_df(df, predictions, model_name):
    newdf = pd.DataFrame(label_encoder.fit_transform(predictions), columns=[model_name])
    df = pd.concat([df, newdf], axis=1)
    return df


print("build predications df")
for i, model in enumerate(model_list):
    model_predictions_df = add_predictions_to_df(model_predictions_df, model[1].predict(tat_test), model[0])
ground_truth = label_encoder.fit_transform(targets_test.to_numpy().ravel())
model_predictions_df = add_predictions_to_df(model_predictions_df, ground_truth, 'ground_truth')


def string_target_to_integer_target(value):
    mapping_dict = {
        'very_high': 0,
        'high': 1,
        'moderate': 2,
        'low': 3,
        'very_low': 4
    }

    return mapping_dict.get(value)


def find_indexes(feature_names, start, end):
    try:
        start_index = feature_names.index(start)
        end_index = feature_names.index(end)
        return start_index, end_index
    except ValueError as e:
        # Handle the case when either start or end value is not found in the list
        print(f"Error: {e}")
        return None, None


def get_data(input_tat, start, end):
    numeric_columns = pd.to_numeric(input_tat.columns, errors='coerce')

    # Filter columns based on the interval
    filtered_columns = input_tat.columns[(numeric_columns >= start) & (numeric_columns <= end)]

    # Create a new DataFrame with only the filtered columns
    input_tat = input_tat[filtered_columns]

    tat_train, tat_test, targets_train, targets_test = train_test_split(input_tat, targets, test_size=0.2,
                                                                        random_state=42)

    targets_train = targets_train.values.ravel()
    targets_test = targets_test.values.ravel()

    feature_names = input_tat.columns.to_list()
    return tat_train, tat_test, targets_train, targets_test, feature_names


def get_model_by_name(model_list, model_name):
    for model_entry in model_list:
        if model_name.lower() in model_entry[0].lower():
            return model_entry
    return None  # Model not found


def get_model_names():
    names = []
    for _, model in enumerate(model_list):
        names.append(model[0])

    return names


def integer_target_to_string_target(value):
    mapping_dict = {
        0: 'very_high',
        1: 'high',
        2: 'moderate',
        3: 'low',
        4: 'very_low'
    }

    return mapping_dict.get(value, 'Unknown')


def model_name_to_key_part_prefix(model_name):
    # last 3 chars of name
    key_part_prefix = model_name[len(model_name) - 5:]
    print(f"Prefix is {key_part_prefix} for {model_name}")
    return key_part_prefix


print("building colors")
ncolor0 = 'rgba(230, 159, 0, 1)'
ncolor1 = 'rgba(0, 158, 115, 1)'
ncolor2 = 'rgba(0, 114, 178, 1)'
ncolor3 = 'rgba(213, 94, 0, 1)'
ncolor4 = 'rgba(204, 121, 167, 1)'

color_node = []

for n in range(4):
    color_node.append(ncolor0)
    color_node.append(ncolor1)
    color_node.append(ncolor2)
    color_node.append(ncolor3)
    color_node.append(ncolor4)

lcolor0 = 'rgba(230, 159, 0, 0.5)'
lcolor1 = 'rgba(0, 158, 115, 0.5)'
lcolor2 = 'rgba(0, 114, 178, 0.5)'
lcolor3 = 'rgba(213, 94, 0, 0.5)'
lcolor4 = 'rgba(204, 121, 167, 0.5)'

color_link = []

for n in range(16):
    color_link.append(lcolor0)
    color_link.append(lcolor1)
    color_link.append(lcolor2)
    color_link.append(lcolor3)
    color_link.append(lcolor4)


def build_sankey(df, model_name_list):
    __NUMBER_OF_TARGETS = 5
    flow_numbers_by_model_indices = {}
    model_names_and_prefixes = list(
        map(lambda model_name: (model_name, model_name_to_key_part_prefix(model_name)), model_name_list)
    )

    # build keys for each model with the next (e.g. if we have 3 models A, B, C we want to see the flow from A to B
    # and from B to C
    for i, current in enumerate(model_names_and_prefixes[:-1]):
        (current_model, current_prefix) = current
        (next_model, next_prefix) = model_names_and_prefixes[i + 1]

        # build keys from model a to b for each combination of values 0 to 4
        for a in range(__NUMBER_OF_TARGETS):
            for b in range(__NUMBER_OF_TARGETS):
                key_part_1 = f"{current_prefix}{a}"
                key_part_2 = f"{next_prefix}{b}"
                key = (key_part_1, key_part_2)

                flow_numbers_by_model_indices[key] = 0

        # count up amount of flow from the dataframe
        for index, row in df.iterrows():
            key_part_1 = f"{current_prefix}{row[current_model]}"
            key_part_2 = f"{next_prefix}{row[next_model]}"
            key = (key_part_1, key_part_2)

            flow_numbers_by_model_indices[key] = flow_numbers_by_model_indices[key] + 1

    # build labels and flow labels
    suffix_labels = []
    model_name_labels = []
    for (model_name, prefix) in model_names_and_prefixes:
        for a in range(__NUMBER_OF_TARGETS):
            suffix_labels.append(f"{prefix}{a}")
            model_name_labels.append(f"{model_name} {integer_target_to_string_target(a)}")

    sources = []
    targets = []
    flow_values = []

    for key in flow_numbers_by_model_indices:
        (source, target) = key
        source_index = suffix_labels.index(source)
        target_index = suffix_labels.index(target)
        value = flow_numbers_by_model_indices[key]
        sources.append(source_index)
        targets.append(target_index)
        flow_values.append(value)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=model_name_labels,
            color=color_node
        ),
        link=dict(
            source=sources,
            target=targets,
            value=flow_values,
            color=color_link
        ))])

    return fig


def generate_new_model(model_entry, tat, start, end):
    tat_train, tat_test, targets_train, targets_test, feature_names = get_data(tat, start, end)

    new_model_name = f"{model_entry[0]} {start}-{end}"
    print(f"building {new_model_name}")
    match model_entry[2]:
        case 'logic_regression':
            model = LogisticRegression(random_state=0)
            model.fit(tat_train, targets_train)

            explainer = shap.Explainer(model, tat_train, feature_names=feature_names)
            explanation = explainer(tat_train)
            model_list.append(
                (
                    new_model_name,
                    model,
                    'logic_regression',
                    explanation
                )
            )

        case 'xgboost':
            use_gpu = False
            targets_train = label_encoder.fit_transform(targets_train)

            if use_gpu:
                model = xgb.XGBClassifier(learning_rate=0.02, n_estimators=10, objective='multi:softmax',
                                          num_class=len(pd.unique(targets_train)), tree_method="hist", device="cuda")
            else:
                model = xgb.XGBClassifier(learning_rate=0.02, n_estimators=10, objective='multi:softmax',
                                          num_class=len(pd.unique(targets_train)))

            model.fit(tat_train, targets_train)

            explainer = shap.TreeExplainer(model, feature_names=feature_names)
            explanation = explainer(tat_train)
            model_list.append(
                (
                    new_model_name,
                    model,
                    'xgboost',
                    explanation
                )
            )

        case 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2, n_jobs=-1)
            model.fit(tat_train, targets_train)
            explainer = shap.TreeExplainer(model, feature_names=feature_names)
            explanation = explainer(tat_train)
            model_list.append(
                (
                    new_model_name,
                    model,
                    'random_forest',
                    explanation
                )
            )
        case _:
            raise Exception('Model is not found.')
    return


app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div(className="container", children=[
    html.Div(className="row", children="Dashboard", style={"textAlign": "center", "color": "black", "fontSize": 30}),

    # erste Zeile Block (Sankey, Retraining)
    html.Div(className="row", children=[
        # Sankey Block
        html.Div(className="ten columns", children=[
            html.Div(className="three columns", children=[
                # Überschrift Sankey
                html.Div(children="Sankey Modelle", style={"textAlign": "left", "color": "black", "fontSize": 25}),
                # Dropdown Sankey
                html.P("Select the models you want to compare...",
                       style={"fontSize": 14, "margin-top": "15px", "color": "grey"}),
                dcc.Dropdown(get_model_names(), placeholder='Select Model 1', id='dropdown_sankey_1'),
                dcc.Dropdown(get_model_names(), placeholder='Select Model 2', id='dropdown_sankey_2'),
                dcc.Dropdown(get_model_names(), placeholder='Select Model 3', id='dropdown_sankey_3'),
                dcc.Dropdown(get_model_names(), placeholder='Select Model 4', id='dropdown_sankey_4'),
                dcc.Checklist(
                    ["Show Ground truth"],
                    id='show_ground_truth_checkbox'
                )
            ]),
            # Diagramm Sankey
            html.Div(className="seven columns", children=[
                html.Div(id='sankey-output-container')
            ])
        ]),  # Sankey Block ENDE

        # TODO Zwischenstrich

        # Retraing Block
        html.Div(className="two columns", children=[
            # Überschrift Retraining
            html.Div(children="Retraining", style={"textAlign": "left", "color": "sankey", "fontSize": 25}),
            html.P("Select the model you want to retrain...",
                   style={"fontSize": 14, "margin-top": "15px", "color": "grey"}),
            dcc.Dropdown(get_model_names(), placeholder='Select Model', id='dropdown_retrain'),
            html.P("Select a data subset by entering start and end wavelength for the new training...",
                   style={"fontSize": 14, "margin-top": "15px", "color": "grey"}),
            dcc.Input(type='number', placeholder='Enter Start', id='input_start_retrain'),
            dcc.Input(type='number', placeholder='Enter End', id='input_end_retrain'),
            html.Br(),
            html.Button('Retrain', id='btn_retrain', style={"margin-top": "25px"}),
            html.Div(id='bb-output-container')
        ])  # Retraing ENDE
    ]),  # ENDE erste Zeile Block (Sankey, Retraining)

    html.Hr(),

    # zweite Zeile Shapley
    html.Div(className="row", children=[
        # Shapley Block
        html.Div(className="four columns", children=[
            # Überschrift Shapley
            html.Div(children="Shapley Values", style={
                "textAlign": "left", "color": "black", "fontSize": 25
            }),
            # Dropdown Shapley
            html.P("Select the model and class of organic soil concentration you want to examine...",
                   style={"fontSize": 14, "margin-top": "15px", "color": "grey"}),
            dcc.Dropdown(get_model_names(), placeholder='Select Model', id='dropdown_model_shapley'),
            dcc.Dropdown(
                [
                    'very_high',
                    'high',
                    'moderate',
                    'low',
                    'very_low'
                ],
                placeholder='Select Target',
                id='dropdown_target'
            ),
            html.P("Enter a range of wavelengths (start and end wavelength)...",
                   style={"fontSize": 14, "margin-top": "15px", "color": "grey"}),
            dcc.Input(type='text', placeholder='Enter Start', id='input_start_shapley'),
            dcc.Input(type='text', placeholder='Enter End', id='input_end_shapley'),
            html.P("Optional: Enter a step number for the wavelengths (e.g. every fourth wavelength = 4)",
                   style={"fontSize": 14, "margin-top": "15px", "color": "grey"}),
            dcc.Input(type='number', placeholder='Enter Steps', id='input_steps'),
            html.Br(),
            html.Button('Generate Plot', id='btn_generate_plot', style={"margin-top": "25px"})
        ]),
        # Diagramm Shapley 1
        html.Div(className="four columns", children=[
            # html.Div(children="Shapley Values", style={"textAlign":"left", "color":"blue", "fontSize":25}),
            html.Div(id='shapley-output-container'),
        ])
        # Diagramm Shapley 2
        # html.Div(children="Shapley Values", style={"textAlign":"left", "color":"blue", "fontSize":25}),
        # Shapley Block ENDE

    ])  # ENDE Zweite Zeile
])


# SANKEY DASHBOARD
@callback(
    Output('sankey-output-container', 'children'),
    [
        Input('dropdown_sankey_1', 'value'),
        Input('dropdown_sankey_2', 'value'),
        Input('dropdown_sankey_3', 'value'),
        Input('dropdown_sankey_4', 'value'),
        Input('show_ground_truth_checkbox', 'value')
    ]
)
def generate_sankey(model_1, model_2, model_3, model_4, show_ground_truth):
    models = [v for v in [model_1, model_2, model_3, model_4] if v is not None]
    if show_ground_truth == ['Show Ground truth']:
        print('adding ground truth')
        models.append('ground_truth')
    return dcc.Graph(figure=build_sankey(model_predictions_df, models))


# SHAPLEY DASHBOARD
@callback(
    Output('shapley-output-container', 'children'),
    State('dropdown_model_shapley', 'value'),  # State instead of Input is important for button
    State('dropdown_target', 'value'),
    State('input_start_shapley', 'value'),
    State('input_end_shapley', 'value'),
    State('input_steps', 'value'),
    Input('btn_generate_plot', 'n_clicks'),
)
def generate_shapley_plots(model, target, start, end, steps, n_clicks):
    if n_clicks is None:
        return dash.no_update  # Do nothing if the button is not clicked

    explanation = get_model_by_name(model_list, model)[3]
    target = string_target_to_integer_target(target)

    if explanation is not None and target is not None:
        fig, ax = plt.subplots()
        ax.cla()  # Clear the previous plot

        start_index, end_index = find_indexes(tat_feature_names, start, end)
        shap.plots.beeswarm(explanation[:, start_index:end_index:steps, target], show=False)

        # Convert Matplotlib figure to base64-encoded string
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='svg')

        plt.clf()  # Clear the entire figure
        plt.close()  # Close the figure to release memory

        img_buf.seek(0)
        img_str = "data:image/svg+xml;base64," + base64.b64encode(img_buf.read()).decode('utf-8')

        return html.Img(src=img_str)
    else:
        return


# RETRAINING DASHBOARD
@callback(
    [
        Output('dropdown_retrain', 'options'),
        Output('dropdown_sankey_1', 'options'),
        Output('dropdown_sankey_2', 'options'),
        Output('dropdown_sankey_3', 'options'),
        Output('dropdown_sankey_4', 'options'),
        Output('dropdown_model_shapley', 'options'),
    ],
    State('dropdown_retrain', 'value'),
    State('input_start_retrain', 'value'),
    State('input_end_retrain', 'value'),
    Input('btn_retrain', 'n_clicks')
)
def retrain_model(model_name, start, end, n_clicks):
    if n_clicks is None:
        return dash.no_update  # Do nothing if the button is not clicked
    if model_name is None or start is None or end is None:
        return dash.no_update  # Do nothing if not all values are provided

    model_entry = get_model_by_name(model_list, model_name)

    _, tat_test, _, _, _ = get_data(tat, start, end)

    generate_new_model(model_entry, tat, start, end)
    print('model has been generated')
    global model_predictions_df
    print('create model predictions for sankey')
    model_predictions_df = add_predictions_to_df(
        model_predictions_df,
        model_list[-1][1].predict(tat_test),
        model_list[-1][0]
    )
    return get_model_names(), get_model_names(), get_model_names(), get_model_names(), get_model_names(), get_model_names()


if __name__ == '__main__':
    app.run(debug=False)
