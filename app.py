import os
import dill

from dash import Dash, html, dcc, Input, Output, State
from src.utils import load_obj

input_features = {
    'fixed acidity': {'placeholder': 'Enter value for fixed acidity', 'min': 4.6, 'max': 15.9, 'step': 0.1},
    'volatile acidity': {'placeholder': 'Enter value for volatile acidity', 'min': 0.12, 'max': 1.58, 'step': 0.01},
    'citric acid': {'placeholder': 'Enter value for citric acid', 'min': 0.0, 'max': 1.0, 'step': 0.1},
    'residual sugar': {'placeholder': 'Enter value for residual sugar', 'min': 0.9, 'max': 15.5, 'step': 0.1},
    'chlorides': {'placeholder': 'Enter value for chlorides', 'min': 0.012, 'max': 0.611, 'step': 0.001},
    'free sulfur dioxide': {'placeholder': 'Enter value for free sulfur dioxide', 'min': 1.0, 'max': 72.0, 'step': 1},
    'total sulfur dioxide': {'placeholder': 'Enter value for total sulfur dioxide', 'min': 6.0, 'max': 289.0, 'step': 1},
    'density': {'placeholder': 'Enter value for density', 'min': 0.99007, 'max': 1.00369, 'step': 0.0001},
    'pH': {'placeholder': 'Enter value for pH', 'min': 2.74, 'max': 4.01, 'step': 0.01},
    'sulphates': {'placeholder': 'Enter value for sulphates', 'min': 0.33, 'max': 2.0, 'step': 0.01},
    'alcohol': {'placeholder': 'Enter value for alcohol', 'min': 8.4, 'max': 14.9, 'step': 0.1},
}

loaded_preprocessor = load_obj(os.path.join("results", "preprocessor.pkl"))
loaded_model = load_obj(os.path.join("results", "model.pkl"))

app = Dash(__name__)

app.layout = html.Div([
    html.Div("Wine Quality Index",
            id = "top-right-corner",
            style = {"float":"right",
                    "padding":"15px",
                    "fontsize":40,
                    "justify-content":"center"}),
    html.Div([html.H1("Wine Quality Predictor",
                      style = {"textAlign":"center",
                               "fontsize": 40,
                               "margin":"0 auto"})],
                        style={"display":"flex",
                               "justify-content":"center"}),
    html.Div([*[dcc.Input(id=key, type='number', placeholder=value['placeholder'], min=value['min'], max=value['max'], step=value['step'])
      for key, value in input_features.items()], html.Button('Get Prediction', id='enter-button', n_clicks=0)]),

    html.Div(id="predicted-output", style={"backgroundColor":"white",
                                       "color":"black",
                                       "display":"flex",
                                       "justify-content":"center",
                                       "height":"13vh",
                                       "align-items":"center"
                                       })
    ], style = {"backgroundColor":"#316395", "color":"white", "margin":"0 auto"})

@app.callback(
    Output("predicted-output", "children"),
    [Input("enter-button", "n_clicks")],
    [State(key, "value") for key in input_features.keys()]
)

def predict_out(n_clicks, *inputs):
    if n_clicks and inputs:
        
        input_array = [[float(value) for value in inputs]]
        preprocessed_arr = loaded_preprocessor.transform(input_array)
        prediction = loaded_model.predict(preprocessed_arr)[0]
        
        return f"Wine Quality: {prediction:.2f}"
    else:
        return "Please enter values for all inputs"
    
if __name__ == '__main__':
    app.run_server(host="0.0.0.0",debug=False)