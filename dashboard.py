from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd  # (versio0.24.2)
import dash, dash_table  # (version 1.0.0)
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.metrics import r2_score
import plotly.express as px
from scipy import stats
import numpy as np

# ----------------------------- Loading Data -------------------------#

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
df = pd.read_csv("TS3_Raw_tree_data.csv")
df = df[df['Age'] > 5]
df = df[df['Leaf (m2)'] > 2]

# ---------------------------------------------------------------------#
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

# -------------------------- app layout -----------------------------#
app.layout = dbc.Container([
    html.Div([
        dbc.Row(
            [
                dbc.Col(html.H4("Dashboard by Jatin Mahour", className="ml-2",
                                style={"font-family": "Times New Roman", 'color': '#3a3733', 'marginTop': 20})),
                dbc.Col(
                    dbc.Button(
                        "Code",
                        href="https://github.com/jatinmahour/Land-use-Classification/blob/main/ag.py",
                        # download="my_data.txt",
                        external_link=True,
                        color="dark",
                        style={"font-family": "Times New Roman", "position": "fixed", "top": 10, "right": 20,
                               "width": 70}
                    )
                ),
            ]
        )
    ]),
    html.Br(),
    html.Div([
        dbc.Row([
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="OV"),
                dbc.Tab(label="Descriptive Analysis", tab_id="DA"),
                dbc.Tab(label="Predictive Analysis", tab_id="PA")
            ], id="tabs")
        ])
    ]),
    html.Div(id="content"),
    html.Div(id="content1"),
    html.Div(id="content2")
])


# ------------------------Overview Tab-------------------------------------------------------#

@app.callback(
    Output(component_id='content', component_property='children'),
    Input(component_id='tabs', component_property='active_tab')
)
def render_tab_content(active_tab):
    if active_tab == "OV":
        return html.Div([
            dbc.Row([
                html.H1("US Urban Tree Growth Analysis",
                        style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733', 'padding': 10}),
                html.Hr(),
                html.H5("This analysis is carried out on urban tree growth data collected over a period of 14 years "
                        "(1998-2012) in 17 cities from 13 states across the United States: Arizona, California, Colorado, "
                        "Florida, Hawaii, Idaho, Indiana, Minnesota, New Mexico, New York, North Carolina, Oregon, and "
                        "South Carolina. Measurements were taken on over 14,000 urban street and park trees. Key information"
                        " collected for each tree species includes bole and crown size, location, and age. The online database "
                        "is available at http://dx.doi.org/10.2737/RDS-2016-0005.",
                        style={"font-family": "HTimes New Roman", 'color': '#3a3733'}, className="ml-2"),
                html.H5("Purpose: Information on urban tree growth underpins models used to calculate effects of trees "
                        "on the environment and human well-being. Maximum tree size and other growth data are used by "
                        "urban forest managers, landscape architects and planners to select trees most suitable to the "
                        "amount of growing space, thereby reducing costly future conflicts between trees and infrastructure."
                        " Growth data are used to develop correlations between growth and influencing factors such as site "
                        "conditions and stewardship practices. "
                        , style={"font-family": "Times New Roman", 'color': '#3a3733'}, className="ml-2")
            ]),
            html.Br(),
            dbc.Row([
                html.Div([
                    html.H4("Urban Tree Sample Dataset"),

                    dash_table.DataTable
                        (
                        df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                        fixed_rows={'headers': True}, fixed_columns={'headers': True},
                        style_table={'height': 200}, style_data={'whiteSpace': 'normal', 'height': 'auto'},
                    )

                ])
            ])
        ])

    # -------------------------------Descriptive Analysis----------------------------------#

    if active_tab == "DA":
        return html.Div([
            html.Div([
                html.Hr(),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Select City", style={"font-family": "Times New Roman", 'color': '#3a3733', 'padding': 5,
                                                  "font-weight": "bold"}),
                    dcc.Dropdown(id='select_city', options=[
                        {'label': x, 'value': x} for x in sorted(df.City.unique())
                    ],
                                 optionHeight=35,  # height/space between dropdown options
                                 value='Berkeley, CA',  # dropdown value selected automatically when page loads
                                 disabled=False,  # disable dropdown value selection
                                 multi=False,  # allow multiple dropdown values to be selected
                                 searchable=True,  # allow user-searching of dropdown values
                                 search_value='',  # remembers the value searched in dropdown
                                 placeholder='Please select...',  # gray, default text shown when no option is selected
                                 clearable=True,  # allow user to removes the selected value
                                 style={'width': "100%"}
                                 ),
                    html.Br(),
                    html.Div([
                        dcc.Graph(id="city",
                                  figure={})
                    ])
                ]),
                dbc.Col([
                    html.Div(dcc.Graph(id="city_map",
                                       figure={}))

                ])
            ]),
            html.Br(),
            html.Hr(),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H5("Select tree species :", style={"font-family": "Times New Roman", "font-weight": "bold",
                                                                'color': '#3a3733',
                                                                'padding': 5}),
                        html.Div([
                            dcc.Dropdown(id='my_dropdown', options=[
                                {'label': 'Sweetgum', 'value': 'Sweetgum'},
                                {'label': 'London planetree ', 'value': 'London planetree'},
                                {'label': 'Northern red oak', 'value': 'Northern red oak'},
                                {'label': 'Silver maple', 'value': 'Silver maple'},
                                {'label': 'Norway maple ', 'value': 'Norway maple'},
                                {'label': 'Honeylocust ', 'value': 'Honeylocust'},
                                {'label': 'Southern magnolia', 'value': 'Southern magnolia'},
                                {'label': 'Red maple', 'value': 'Red maple'},
                                {'label': 'Green ash ', 'value': 'Green ash'},
                                {'label': 'Camphor tree ', 'value': 'Camphor tree'},
                                {'label': 'Siberian elm   ', 'value': 'Siberian elm'},
                                {'label': 'Callery pear', 'value': 'Callery pear'},
                                {'label': 'Sugar maple ', 'value': 'Sugar maple'},
                                {'label': 'American elm   ', 'value': 'American elm'},
                                {'label': 'Ginkgo', 'value': 'Ginkgo'},
                                {'label': 'Northern hackberry', 'value': 'Northern hackberry'},
                                {'label': 'Chinese pistache  ', 'value': 'Chinese pistache'},
                                {'label': 'Common crapemyrtle  ', 'value': 'Common crapemyrtle'},
                                {'label': 'Littleleaf linden   ', 'value': 'Littleleaf linden'},
                                {'label': 'White ash ', 'value': 'White ash'}

                            ],
                                         optionHeight=35,  # height/space between dropdown options
                                         value='Red maple',  # dropdown value selected automatically when page loads
                                         disabled=False,  # disable dropdown value selection
                                         multi=False,  # allow multiple dropdown values to be selected
                                         searchable=True,  # allow user-searching of dropdown values
                                         search_value='',  # remembers the value searched in dropdown
                                         placeholder='Please select...',
                                         # gray, default text shown when no option is selected
                                         clearable=True,  # allow user to removes the selected value
                                         style={'width': "100%"}),
                            html.Br(),
                            dcc.Graph(id="graph1",
                                      figure={})
                        ])
                    ]),
                    dbc.Col([
                        html.Div(id="red maple")
                    ])
                ])
            ]),
            html.Br(),
            html.Hr(),
            html.Div([
                html.H5(
                    "Histogram of the age-wise distribution for the selected tree species. "
                    "Hovering over the boxplot, shows the values of median and other quartiles.",
                    style={"font-family": "Times New Roman", "font-weight": "bold", 'color': '#3a3733',
                           "text-align": "center",
                           'padding': 5
                           }),
                dcc.Graph(id="graph2",
                          figure={})
            ]),
            html.Hr(),
            html.Div([
                html.H5("This chart shows the distribution of tree heights.",
                        style={"font-family": "Times New Roman", "font-weight": "bold", "text-align": "center",
                               'color': '#3a3733', 'padding': 5}),
                dcc.Graph(id="graph3",
                          figure={})
            ]),
            html.Hr(),
            html.Div([
                html.H5(
                    "This chart shows the distribution of DBH(Diameter at breast height).",
                    style={"font-family": "Times New Roman", "text-align": "center", "font-weight": "bold",
                           'color': '#3a3733',
                           'padding': 5}),
                dcc.Graph(id="graph4",
                          figure={})
            ]),
            html.Div([
                html.Hr(),
                html.H5(
                    "This is the scatterplot matrix plotted between variables of interest. Here we get the glimpse of the corelation in variables.",
                    style={"font-family": "Times New Roman", "font-weight": "bold", "text-align": "center",
                           'color': '#3a3733',
                           'padding': 5}),
                dcc.Graph(id="scatter-matrix",
                          figure={})
            ])

        ])

    # -------------------------------- Predictive Analysis ----------------------------------#

    if active_tab == "PA":
        return html.Div([
            dbc.Row([
                html.H5("Predict Leaf area (m2) : Multiple Linear Regression"
                        , style={"font-family": "Times New Roman", 'padding': 10, 'marginTop': 10}),
                dbc.Col([
                    html.P("Select Tree Species :", style={'padding': 5, "font-weight": "bold"}),
                    dcc.Dropdown(id='my_dropdown1', options=[
                        {'label': 'Sweetgum', 'value': 'Sweetgum'},
                        {'label': 'London planetree ', 'value': 'London planetree'},
                        {'label': 'Northern red oak', 'value': 'Northern red oak'},
                        {'label': 'Silver maple', 'value': 'Silver maple'},
                        {'label': 'Norway maple ', 'value': 'Norway maple'},
                        {'label': 'Honeylocust ', 'value': 'Honeylocust'},
                        {'label': 'Southern magnolia', 'value': 'Southern magnolia'},
                        {'label': 'Red maple', 'value': 'Red maple'},
                        {'label': 'Green ash ', 'value': 'Green ash'},
                        {'label': 'Camphor tree ', 'value': 'Camphor tree'},
                        {'label': 'Siberian elm   ', 'value': 'Siberian elm'},
                        {'label': 'Callery pear', 'value': 'Callery pear'},
                        {'label': 'Sugar maple ', 'value': 'Sugar maple'},
                        {'label': 'American elm   ', 'value': 'American elm'},
                        {'label': 'Ginkgo', 'value': 'Ginkgo'},
                        {'label': 'Northern hackberry', 'value': 'Northern hackberry'},
                        {'label': 'Chinese pistache  ', 'value': 'Chinese pistache'},
                        {'label': 'Common crapemyrtle  ', 'value': 'Common crapemyrtle'},
                        {'label': 'Littleleaf linden   ', 'value': 'Littleleaf linden'},
                        {'label': 'White ash ', 'value': 'White ash'}

                    ],
                                 optionHeight=35,  # height/space between dropdown options
                                 value='Red maple',  # dropdown value selected automatically when page loads
                                 disabled=False,  # disable dropdown value selection
                                 multi=False,  # allow multiple dropdown values to be selected
                                 searchable=True,  # allow user-searching of dropdown values
                                 search_value='',  # remembers the value searched in dropdown
                                 placeholder='Please select...',
                                 # gray, default text shown when no option is selected
                                 clearable=True,  # allow user to removes the selected value
                                 style={'width': "70%"})
                ], width=4),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Row(
                                html.Label('Age', style={'padding': 10})
                            ),
                            dbc.Row(
                                dcc.Input(id='var1', type='number', value=20, style={'width': "70%"})
                            )]),
                        dbc.Col([
                            dbc.Row(
                                html.Label('Tree Height (m)', style={'padding': 10})
                            ),
                            dbc.Row(
                                dcc.Input(id='var2', type='number', value=17, style={'width': "70%"})
                            )]),
                        dbc.Col([
                            dbc.Row(
                                html.Label('DBH (cm)', style={'padding': 10})
                            ),
                            dbc.Row(
                                dcc.Input(id='var3', type='number', value=65, style={'width': "70%"})
                            )])
                    ]),
                    html.Br(),
                    html.Div(id='output-prediction', style={'padding': 10, "font-weight": "bold"})
                ]),
                dbc.Row([
                    html.Div([
                        html.Hr(),
                        dcc.Graph(id="Reg2",
                                  figure={}),
                        html.Hr(),
                        html.H5("Enhanced prediction error analysis: Visualize how well your model generalizes"
                                " by comparing it with the theoretical optimal fit (black dotted line)"),
                        dcc.Graph(id="Reg3",
                                  figure={}),
                        html.Hr(),
                        html.H5("Residual plots:  visualize your prediction residuals "),
                        dcc.Graph(id="Reg4",
                                  figure={})
                    ])
                ])
            ])
        ])


# ---------------------------------------- Leaf Area prediction --------------------------#

@app.callback(
    Output('output-prediction', 'children'),
    [Input(component_id='my_dropdown1', component_property='value')],
    # [Input('predict-button', 'n_clicks')],
    [Input('var1', 'value'), Input('var2', 'value'),
     Input('var3', 'value')]
)
def update_output(value_chosen, var1, var2, var3):
    dff = df[df["CommonName"] == value_chosen]
    dff['Leaf (m2)'], _ = stats.boxcox(dff['Leaf (m2)'])
    dff['DBH (cm)'], _ = stats.boxcox(dff['DBH (cm)'])
    dff['TreeHt (m)'], _ = stats.boxcox(dff['TreeHt (m)'])
    dff['Age'], _ = stats.boxcox(dff['Age'])

    X = dff[["Age", 'TreeHt (m)', "DBH (cm)"]].values
    y = dff['Leaf (m2)'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

    model = LinearRegression()
    model.fit(X_train, y_train)

    input_data = np.array([[var1, var2, var3]])
    predicted_value = model.predict(input_data)
    r_squared = r2_score(y_train, model.predict(X_train))

    results = f"Predicted Value (Leaf Area): {predicted_value[0]:.4f}\n\n"
    results += f"R-squared: {r_squared:.4f}"

    return html.P(results)


# --------------------------------- Features Analysis graph --------------------------#

@app.callback(
    Output(component_id='Reg2', component_property='figure'),
    Input(component_id='my_dropdown1', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]
    dfff = dff[['Age', 'TreeHt (m)', "DBH (cm)", "TreeType"]]
    dff['Leaf (m2)'], _ = stats.boxcox(dff['Leaf (m2)'])
    dfff['DBH (cm)'], _ = stats.boxcox(dfff['DBH (cm)'])
    dfff['TreeHt (m)'], _ = stats.boxcox(dfff['TreeHt (m)'])
    dfff['Age'], _ = stats.boxcox(dfff['Age'])
    X = dfff.drop(columns=["TreeType"])
    y = dff['Leaf (m2)']

    model = LinearRegression()
    model.fit(X, y)
    colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]
    fig = px.bar(
        x=X.columns, y=model.coef_, color=colors,
        color_discrete_sequence=['red', 'blue'],
        labels=dict(x='Feature', y='Linear coefficient'),
        title='Weight of each feature for predicting Leaf Area'
    )
    return fig


# ------------------------ Prediction Error Graph -----------------------------------#

@app.callback(
    Output(component_id='Reg3', component_property='figure'),
    Input(component_id='my_dropdown1', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]
    # Split data into training and test splits
    train_idx, test_idx = train_test_split(dff.index, test_size=.25, random_state=0)
    dff['split'] = 'train'
    dff.loc[test_idx, 'split'] = 'test'
    dff['Leaf (m2)'], _ = stats.boxcox(dff['Leaf (m2)'])
    dff['DBH (cm)'], _ = stats.boxcox(dff['DBH (cm)'])
    dff['TreeHt (m)'], _ = stats.boxcox(dff['TreeHt (m)'])
    dff['Age'], _ = stats.boxcox(dff['Age'])

    X = dff[["Age", 'TreeHt (m)', "DBH (cm)"]]
    y = dff['Leaf (m2)']
    X_train = dff.loc[train_idx, ["Age", 'TreeHt (m)', "DBH (cm)"]]
    y_train = dff.loc[train_idx, 'Leaf (m2)']

    model = LinearRegression()
    model.fit(X_train, y_train)
    dff['prediction'] = model.predict(X)

    fig = px.scatter(
        dff, x='Leaf (m2)', y='prediction',
        marginal_x='histogram', marginal_y='histogram',
        color='split', trendline='ols'
    )

    fig.update_traces(histnorm='probability', selector={'type': 'histogram'})
    fig.add_shape(
        type="line", line=dict(dash='dash'),
        x0=y.min(), y0=y.min(),
        x1=y.max(), y1=y.max()
    )

    return fig


# ----------------------------- Residual Analysis graph -----------------------------------#

@app.callback(
    Output(component_id='Reg4', component_property='figure'),
    Input(component_id='my_dropdown1', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]
    # Split data into training and test splits
    train_idx, test_idx = train_test_split(dff.index, test_size=.25, random_state=0)
    dff['split'] = 'train'
    dff.loc[test_idx, 'split'] = 'test'
    dff['Leaf (m2)'], _ = stats.boxcox(dff['Leaf (m2)'])
    dff['DBH (cm)'], _ = stats.boxcox(dff['DBH (cm)'])
    dff['TreeHt (m)'], _ = stats.boxcox(dff['TreeHt (m)'])
    dff['Age'], _ = stats.boxcox(dff['Age'])

    X = dff[["Age", 'TreeHt (m)', "DBH (cm)"]]
    # y = dff['Leaf (m2)']
    X_train = dff.loc[train_idx, ["Age", 'TreeHt (m)', "DBH (cm)"]]
    y_train = dff.loc[train_idx, 'Leaf (m2)']

    model = LinearRegression()
    model.fit(X_train, y_train)
    dff['prediction'] = model.predict(X)
    dff['residual'] = dff['prediction'] - dff['Leaf (m2)']

    fig = px.scatter(
        dff, x='prediction', y='residual',
        marginal_y='violin',
        color='split', trendline='ols'
    )

    return fig


# ---------------------------- Bargraph for city ---------------------------------------#

@app.callback(
    Output(component_id='city', component_property='figure'),
    Input(component_id='select_city', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["City"] == value_chosen]
    fig = px.histogram(
        data_frame=dff,
        x='CommonName',
        color='CommonName',
        hover_data=['ScientificName']
    )
    fig.update_layout(title="Count of distinct tree species in the selected city.",
                      showlegend=False, xaxis_showticklabels=False, title_font_family="Times New Roman")
    return fig


# --------------------------- Bargraph for tree species ---------------------------------#

@app.callback(
    Output(component_id='graph1', component_property='figure'),
    Input(component_id='my_dropdown', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]
    fig = px.histogram(
        data_frame=dff,
        x='City',
        color='City'
    )
    fig.update_layout(title="Count of selected tree species in each city.",
                      title_font_family="Times New Roman")
    return fig


# ------------------------ Map USA -----------------------------------------#

@app.callback(
    Output(component_id='city_map', component_property='figure'),
    Input(component_id='select_city', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["City"] == value_chosen]
    fig = px.scatter_mapbox(dff, lat="Lat", lon="Long", hover_name="City", hover_data=["Region"],
                            color_discrete_sequence=["fuchsia"], zoom=3, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig


# -------------------------- Histogram Age-wise ----------------------------------------#

@app.callback(
    Output(component_id='graph2', component_property='figure'),
    Input(component_id='my_dropdown', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]

    fig = px.histogram(
        data_frame=dff,
        x='Age',
        color='City',
        marginal="box",
        histnorm='probability density',
        facet_col='City',
        nbins=10)

    return fig


# ------------------------------------- Histogram tree-ht wise ------------------------------------#

@app.callback(
    Output(component_id='graph3', component_property='figure'),
    Input(component_id='my_dropdown', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]

    fig = px.histogram(
        data_frame=dff,
        x='TreeHt (m)',
        color='City',
        marginal="box",
        histnorm='probability density',
        facet_col='City',
        barmode='stack')

    return fig


# ----------------------------------------- Histogram DBH-wise ----------------------------------#

@app.callback(
    Output(component_id='graph4', component_property='figure'),
    Input(component_id='my_dropdown', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]

    fig = px.histogram(
        data_frame=dff,
        x='DBH (cm)',
        color='City',
        marginal="box",
        histnorm='probability density',
        facet_col='City',
        barmode='stack')

    return fig


# ------------------------------------------- Scatter Matrix ---------------------------#


@app.callback(
    Output(component_id='scatter-matrix', component_property='figure'),
    Input(component_id='my_dropdown', component_property='value')
)
def update_graph(value_chosen):
    dff = df[df["CommonName"] == value_chosen]

    fig = px.scatter_matrix(dff,
                            dimensions=["Age", "DBH (cm)", "TreeHt (m)", "AvgCdia (m)", "Leaf (m2)"],
                            color="City", width=1300, height=700)

    return fig


# ------------------------------------------------ Tree description ------------------------------------#

@app.callback(Output('red maple', 'children'),
              Input('my_dropdown', 'value'))
def update_image(value_chosen):
    if value_chosen == "Red maple":
        return html.Div([html.Img(src=app.get_asset_url('RM.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Acer rubrum, the red maple, also known as swamp, water or soft maple, is one of the most "
                             "common and widespread deciduous trees of eastern and central North America. The U.S. Forest service"
                             " recognizes it as the most abundant native tree in eastern North America.[4] The red maple ranges "
                             "from southeastern Manitoba around the Lake of the Woods on the border with Ontario and Minnesota, "
                             "east to Newfoundland, south to Florida, and southwest to East Texas. Many of its features, "
                             "especially its leaves, are quite variable in form. At maturity, it often attains a height of "
                             "around 30 m (100 ft). Its flowers, petioles, twigs and seeds are all red to varying degrees. "
                             "Among these features, however, it is best known for its brilliant deep scarlet foliage in autumn.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Red Maple', href='https://en.wikipedia.org/wiki/Acer_rubrum')
                         ]),
    if value_chosen == "London planetree":
        return html.Div([html.Img(src=app.get_asset_url('LP.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Platanus × acerifolia, Platanus × hispanica, or hybrid plane, "
                             "is a tree in the genus Platanus. It is often known by the synonym London plane,[2]"
                             " or London planetree. It is a hybrid of Platanus orientalis (oriental plane) and"
                             " Platanus occidentalis (American sycamore).",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-London Planetree',
                                  href='https://en.wikipedia.org/wiki/Platanus_%C3%97_acerifolia')
                         ]),
    if value_chosen == "Sweetgum":
        return html.Div([html.Img(src=app.get_asset_url('SG.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "American sweetgum (Liquidambar styraciflua), "
                             "also known as American storax,[3] hazel pine,[4] bilsted,[5] redgum,[3] "
                             "satin-walnut,[3] star-leaved gum,[5] alligatorwood,[3] or simply sweetgum,"
                             "[3][6] is a deciduous tree in the genus Liquidambar native to warm temperate "
                             "areas of eastern North America and tropical montane regions of Mexico and Central"
                             " America. Sweetgum is one of the main valuable forest trees in the southeastern "
                             "United States, and is a popular ornamental tree in temperate climates. "
                             "It is recognizable by the combination of its five-pointed star-shaped leaves "
                             "(similar to maple leaves) and its hard, spiked fruits. "
                             "It is currently classified in the plant family Altingiaceae, "
                             "but was formerly considered a member of the Hamamelidaceae.[7]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Sweetgum', href='https://en.wikipedia.org/wiki/Liquidambar_styraciflua')
                         ]),
    if value_chosen == "Northern red oak":
        return html.Div([html.Img(src=app.get_asset_url('NRO.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Quercus rubra, the northern red oak, is an oak tree in the"
                             " red oak group (Quercus section Lobatae). It is a native of North America, "
                             "in the eastern and central United States and southeast and south-central Canada."
                             " It has been introduced to small areas in Western Europe, where it can frequently"
                             " be seen cultivated in gardens and parks. It prefers good soil that is slightly acidic. "
                             "Often simply called red oak, northern red oak is so named to distinguish it from southern red oak"
                             " (Q. falcata), also known as the Spanish oak. Northern Red Oak is sometimes called champion oak.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Northern red oak', href='https://en.wikipedia.org/wiki/Quercus_rubra')
                         ]),
    if value_chosen == "Silver maple":
        return html.Div([html.Img(src=app.get_asset_url('SM.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Acer saccharinum, commonly known as silver maple,[3]"
                             " creek maple, silverleaf maple,[3] soft maple, large maple,[3] water maple,[3]"
                             " swamp maple,[3] or white maple,[3] is a species of maple native to the eastern"
                             " and central United States and southeastern Canada.[3][4] It is one of the most"
                             " common trees in the United States." "Although the silver maple's Latin name"
                             " is similar, it should not be confused with Acer saccharum, the sugar maple. "
                             "Some of the common names are also applied to other maples, especially Acer rubrum.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Silver maple', href='https://en.wikipedia.org/wiki/Acer_saccharinum')
                         ]),
    if value_chosen == "Norway maple":
        return html.Div([html.Img(src=app.get_asset_url('NM.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Acer platanoides, commonly known as the Norway maple, is a"
                             " species of maple native to eastern and central Europe and western Asia,"
                             " from Spain east to Russia, north to southern Scandinavia and southeast to"
                             " northern Iran.[2][3][4] It was introduced to North America in the mid-1700s as "
                             "a shade tree.[5] It is a member of the family Sapindaceae.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Norway maple', href='https://en.wikipedia.org/wiki/Acer_platanoides')
                         ]),
    if value_chosen == "Honeylocust":
        return html.Div([html.Img(src=app.get_asset_url('HL.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "The honey locust (Gleditsia triacanthos), "
                             "also known as the thorny locust or thorny honeylocust, is a "
                             "deciduous tree in the family Fabaceae, native to central North America "
                             "where it is mostly found in the moist soil of river valleys.[3]"
                             " Honey locust is highly adaptable to different environments, has been "
                             "introduced worldwide, and can be an aggressive, invasive species outside of "
                             "its native range.[3]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Honeylocust', href='https://en.wikipedia.org/wiki/Honey_locust')
                         ]),
    if value_chosen == "Southern magnolia":
        return html.Div([html.Img(src=app.get_asset_url('SMG.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Southern magnolia, commonly known as the southern"
                             " magnolia or bull bay, is a tree of the family Magnoliaceae "
                             "native to the Southeastern United States, from Virginia to central Florida, "
                             "and west to East Texas.[5] Reaching 27.5 m (90 ft) in height, it is a large, "
                             "striking evergreen tree, with large, dark-green leaves up to 20 cm (7+3⁄4 in) "
                             "long and 12 cm (4+3⁄4 in) wide, and large, white, fragrant flowers up to 30 cm (12 in) "
                             "in diameter." "Although endemic to the evergreen lowland subtropical forests on "
                             "the Gulf and South Atlantic coastal plain, M. grandiflora is widely "
                             "cultivated in warmer areas around the world. The timber is hard and heavy, "
                             "and has been used commercially to make furniture, pallets, and veneer.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Southern magnolia',
                                  href='https://en.wikipedia.org/wiki/Magnolia_grandiflora')
                         ]),
    if value_chosen == "Green ash":
        return html.Div([html.Img(src=app.get_asset_url('GA.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Fraxinus pennsylvanica, the green ash or red ash,[2] "
                             "is a species of ash native to eastern and central North America, "
                             "from Nova Scotia west to southeastern Alberta and eastern Colorado, "
                             "south to northern Florida, and southwest to Oklahoma and eastern Texas. "
                             "It has spread and become naturalized in much of the western United States "
                             "and also in Europe from Spain to Russia.[3][4][5]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Green ash', href='https://en.wikipedia.org/wiki/Fraxinus_pennsylvanica')
                         ]),
    if value_chosen == "Camphor tree":
        return html.Div([html.Img(src=app.get_asset_url('CT.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Camphora officinarum is a species of evergreen "
                             "tree that is commonly known under the names camphor tree,"
                             " camphorwood or camphor laurel.[1][2]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Camphor tree', href='https://en.wikipedia.org/wiki/Camphora_officinarum')
                         ]),
    if value_chosen == "Siberian elm":
        return html.Div([html.Img(src=app.get_asset_url('SE.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Ulmus pumila, the Siberian elm, is a tree native to Asia. "
                             "It is also known as the Asiatic elm and dwarf elm, but sometimes miscalled"
                             " the 'Chinese elm' (Ulmus parvifolia). U. pumila has been widely cultivated "
                             "throughout Asia, North America, Argentina, and southern Europe, becoming "
                             "naturalized in many places, notably across much of the United States.[2][3]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Siberian elm', href='https://en.wikipedia.org/wiki/Ulmus_pumila')
                         ]),
    if value_chosen == "Callery pear":
        return html.Div([html.Img(src=app.get_asset_url('CP.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Pyrus calleryana, or the Callery pear,"
                             " is a species of pear tree native to China and"
                             " Vietnam,[2] in the family Rosaceae. It is most "
                             "commonly known for its cultivar 'Bradford' and its "
                             "offensive odor, widely planted throughout the United States"
                             " and increasingly regarded as an invasive species.[2]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Siberian elm', href='https://en.wikipedia.org/wiki/Pyrus_calleryana')
                         ]),
    if value_chosen == "Sugar maple":
        return html.Div([html.Img(src=app.get_asset_url('SUM.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Acer saccharum, the sugar maple, is a species of flowering "
                             "plant in the soapberry and lychee family Sapindaceae. "
                             "It is native to the hardwood forests of eastern Canada"
                             " and eastern United States.[3] Sugar maple is best known "
                             "for being the primary source of maple syrup and for its"
                             " brightly colored fall foliage.[4]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Sugar maple', href='https://en.wikipedia.org/wiki/Acer_saccharum')
                         ]),
    if value_chosen == "American elm":
        return html.Div([html.Img(src=app.get_asset_url('AE.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Ulmus americana, generally known as the American elm or,"
                             " less commonly, as the white elm or water elm,[a] is a species of"
                             " elm native to eastern North America, naturally occurring from Nova "
                             "Scotia west to Alberta and Montana, and south to Florida and central Texas."
                             " The American elm is an extremely hardy tree that can withstand winter "
                             "temperatures as low as −42 °C (−44 °F). Trees in areas unaffected by Dutch "
                             "elm disease (DED) can live for several hundred years.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-American elm', href='https://en.wikipedia.org/wiki/Ulmus_americana')
                         ]),
    if value_chosen == "Ginkgo":
        return html.Div([html.Img(src=app.get_asset_url('GK.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Ginkgo biloba, commonly known as ginkgo or gingko"
                             " (/ˈɡɪŋkoʊ, ˈɡɪŋkɡoʊ/ GINK-oh, -⁠goh),[5][6] also known as the"
                             " maidenhair tree,[7] is a species of gymnosperm tree native to East Asia. "
                             "It is the last living species in the order Ginkgoales, which first appeared "
                             "over 290 million years ago. Fossils very similar to the living species, "
                             "belonging to the genus Ginkgo, extend back to the Middle Jurassic epoch "
                             "approximately 170 million years ago.[2] The tree was cultivated early in"
                             " human history and remains commonly planted.Ginkgo leaf extract is commonly "
                             "used as a dietary supplement, but there is no scientific evidence that it "
                             "supports human health or is effective against any disease.[8][9]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Ginkgo', href='https://en.wikipedia.org/wiki/Ginkgo_biloba')
                         ]),
    if value_chosen == "Northern hackberry":
        return html.Div([html.Img(src=app.get_asset_url('GK.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Celtis occidentalis, commonly known as the common hackberry,"
                             " is a large deciduous tree native to North America."
                             " It is also known as the nettletree, sugarberry, beaverwood,"
                             " northern hackberry, and American hackberry.[4]"
                             " It is a moderately long-lived[4] hardwood[4] with a light-colored wood, "
                             "yellowish gray to light brown with yellow streaks.[5]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Northern hackberry',
                                  href='https://en.wikipedia.org/wiki/Celtis_occidentalis')
                         ]),
    if value_chosen == "Chinese pistache":
        return html.Div([html.Img(src=app.get_asset_url('CPT.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Pistacia chinensis, the Chinese pistache[3] (Chinese: 黄連木; pinyin: huángliánmù),"
                             " is a small to medium-sized tree in the genus Pistacia in the cashew family Anacardiaceae, "
                             "native to central and western China.[4] This species is planted as a street tree in temperate"
                             " areas worldwide due to its attractive fruit and autumn foliage.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Chinese pistache', href='https://en.wikipedia.org/wiki/Pistacia_chinensis')
                         ]),
    if value_chosen == "Common crapemyrtle":
        return html.Div([html.Img(src=app.get_asset_url('CC.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Lagerstroemia indica, the crape myrtle (also crepe myrtle, crêpe myrtle, "
                             "or crepeflower[1]) is a species of flowering plant in the genus Lagerstroemia of"
                             " the family Lythraceae. It is native to the Indian Subcontinent (hence the species"
                             " epithet indica), and also to Southeast Asia, China, Korea and Japan. The genus name "
                             "honors Swedish botanist Magnus von Lagerström.[2] It is an often multi-stemmed, "
                             "deciduous tree with a wide spreading, flat topped, rounded, or even spike shaped open "
                             "habit. The tree is a popular nesting shrub for songbirds and wrens.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Common crapemyrtle',
                                  href='https://en.wikipedia.org/wiki/Lagerstroemia_indica')
                         ]),
    if value_chosen == "Littleleaf linden":
        return html.Div([html.Img(src=app.get_asset_url('LL.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Tilia cordata, the small-leaved lime or small-leaved linden, "
                             "is a species of tree in the family Malvaceae, native to much of Europe. "
                             "Other common names include little-leaf or littleleaf linden,[2] or traditionally "
                             "in South East England, pry or pry tree.[3] Its range extends from Britain through"
                             " mainland Europe to the Caucasus and western Asia. In the south of its range it is"
                             " restricted to high elevations.[4][5]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Littleleaf linden', href='https://en.wikipedia.org/wiki/Tilia_cordata')
                         ]),
    if value_chosen == "Littleleaf linden":
        return html.Div([html.Img(src=app.get_asset_url('LL.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Tilia cordata, the small-leaved lime or small-leaved linden, "
                             "is a species of tree in the family Malvaceae, native to much of Europe. "
                             "Other common names include little-leaf or littleleaf linden,[2] or traditionally "
                             "in South East England, pry or pry tree.[3] Its range extends from Britain through"
                             " mainland Europe to the Caucasus and western Asia. In the south of its range it is"
                             " restricted to high elevations.[4][5]",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-Littleleaf linden', href='https://en.wikipedia.org/wiki/Tilia_cordata')
                         ]),
    if value_chosen == "White ash":
        return html.Div([html.Img(src=app.get_asset_url('WA.jpg'), height="300px", width="300px",
                                  style={'marginTop': 20, 'padding': 20}, className="ml-2"
                                  ),
                         html.H6(
                             "Fraxinus americana, the white ash or American ash, is a species of ash "
                             "tree native to eastern and central North America. The species is native to "
                             "mesophytic hardwood forests from Nova Scotia west to Minnesota, south to northern "
                             "Florida, and southwest to eastern Texas. Isolated populations have also been found "
                             "in western Texas, Wyoming, and Colorado, and the species is reportedly naturalized in "
                             "Hawaii.[3][4][5].There are an estimated 8 billion ash trees in the United"
                             " States[citation needed] – the majority being the white ash trees and the green ash trees.",
                             style={"font-family": "Times New Roman", 'marginTop': 20, 'color': '#3a3733',
                                    'padding': 20}, className="ml-2"),
                         dcc.Link('Wikipedia-White ash', href='https://en.wikipedia.org/wiki/Fraxinus_americana')
                         ])


# ----------------------------------------- End --------------------------------------------------------------#

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
