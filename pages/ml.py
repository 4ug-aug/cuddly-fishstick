import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# import pytorch
import torch
import torch.nn as nn
# plotly
import plotly.express as px
from torchsummary import summary
from stqdm import stqdm

st.set_page_config(
        page_title="Machine Learning",
        page_icon="🧊",
)

st.title('Machine Learning')
st.sidebar.subheader('Model design')

#....................... DATA ACCESS .......................
if st.session_state.get("ml_data") is not None:
    df_ml = st.session_state.ml_data
    st.write("""*A machine learning dataset has been created based on the uploaded data. 
                You can now select the features and target variable, and design the model without altering your original data.*""")
    if st.button("Restore data"):
        st.session_state.ml_data = None
        st.rerun()
    st.write(df_ml.head(5))

elif st.session_state.get("data") is not None:
    df_ml = st.session_state.data
    st.session_state.ml_data = df_ml
    st.write("""*A machine learning dataset has been created based on the uploaded data. 
                You can now select the features and target variable, and design the model without altering your original data.*""")
    st.write(df_ml.head(5))
else:
    st.warning("No data uploaded")
    st.stop()

def get_model(model_name):
    if model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'SVM':
        model = SVC()
    else:
        model = None
    return model

def add_drop_layer(network, p=0.5):
    layer_n = st.session_state.layer_n + 1
    network.add_module(f'layer_{layer_n}', nn.Dropout(p))
    st.session_state.layer_n = layer_n
    return network

def add_neural_layer(network, in_features=100, out_features=100, activation='ReLU'):
    layer_n = st.session_state.layer_n + 1
    st.toast(f"Added layer {layer_n}, with input size {in_features} and output size {out_features}")
    network.add_module(f'layer_{layer_n}', nn.Linear(in_features, out_features))
    network.add_module(f'activation_{layer_n}', activation_map[activation])
    
    st.session_state.prev_out_size = out_features

    st.session_state.layer_n = layer_n
    return network

# Select model
model_name = st.sidebar.selectbox('Select model', 
                                  ['Random Forest', 
                                   'SVM',
                                   'Neural Network'])

# Deselect features
removed_features = st.sidebar.multiselect('Deselect features', 
                                  df_ml.columns.tolist())

# Remove features
if len(removed_features) > 0:
    df_ml = df_ml.drop(removed_features, axis=1)
    st.session_state.ml_data = df_ml

# Select target
target = st.sidebar.selectbox('Select target', 
                                  df_ml.columns.tolist())

# @st.cache
def train_neural_network(network, X_train, y_train, epochs=10, lr=0.001):
    # check target type
    if df_ml[target].dtype == 'object':
        criterion = nn.CrossEntropyLoss()
        # convert target to one-hot encoding
        y_train = torch.tensor(pd.get_dummies(y_train).values).float()
        X_train = torch.tensor(X_train).float()

        # check final layer output size (number of categories)
        if st.session_state.prev_out_size != len(df_ml[target].unique()):
            st.write('Adding final layer, in_features = ', st.session_state.prev_out_size, 'out_features = ', len(df_ml[target].unique()))
            network.add_module(f'layer_{len(network)}', nn.Linear(st.session_state.prev_out_size, len(df_ml[target].unique())))
            network.add_module(f'activation_{len(network)}', nn.Softmax())
            st.write(network)
    else:
        criterion = nn.MSELoss()
        y_train = torch.tensor(y_train.values).float()

    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    st.write(f"Shape of X_train: {X_train.shape}")

    for epoch in stqdm(range(epochs), desc="Training neural network"):
        optimizer.zero_grad()
        outputs = network(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # matplotlib plot of loss
            fig = px.line(x=range(epoch), y=loss)
            st.plotly_chart(fig, use_container_width=True)
    return network

# Design model
test_size = st.sidebar.slider('Test size', 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.slider('Random state', 0, 100, 42, 1)

if model_name == 'Random Forest':
    n_estimators = st.sidebar.slider('Number of estimators', 1, 1000, 100, 1)
    max_depth = st.sidebar.slider('Max depth', 1, 100, 10, 1)

    # if the target is a category, use the classifier
    if df_ml[target].dtype == 'object':
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                       max_depth=max_depth, 
                                       random_state=random_state,
                                       verbose=1)

    # if the target is numeric, use the regressor
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                      max_depth=max_depth, 
                                      random_state=random_state,
                                      verbose=1)
    
elif model_name == 'SVM':
    C = st.sidebar.slider('C', 0.01, 10.0, 1.0, 0.01)
    if df_ml[target].dtype == 'object':
        model = SVC(C=C, random_state=random_state, verbose=1)
    else:
        st.sidebar.warning('SVM is not suitable for numeric targets')

elif model_name == 'Neural Network':
    st.subheader('Neural Network')
    
    st.sidebar.subheader('Layers')
    if st.session_state.get("network") is None:
        network = nn.Sequential()
        st.session_state.network = network
    else:
        network = st.session_state.network
    if st.session_state.get("layer_n") is None:
        st.session_state.layer_n = 0

    nn_model_col1, nn_model_col2, nn_model_col3 = st.sidebar.columns(3)
    
    if st.session_state.layer_n == 0:
        in_features = int((df_ml.shape[0] * (1-test_size)))
        st.session_state.prev_out_size = in_features
    else:
        # in_features is the current number of features
        in_features = st.session_state.prev_out_size
        # pass

    layer_type = st.sidebar.selectbox('Layer type', ['Dense', 'Dropout'])
    if layer_type == 'Dropout':
        p = st.sidebar.slider('Dropout probability', 0.0, 1.0, 0.5, 0.05)
        if nn_model_col1.button('Add layer'):
            network = add_drop_layer(network, p)
            st.session_state.network = network

    elif layer_type == 'Dense':
        out_features = st.sidebar.slider('Units', 1, 126, 100, 1)
        activation = st.sidebar.selectbox('Activation', ['relu', 'sigmoid', 'tanh'])
        activation_map = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
        if nn_model_col1.button('Add layer'):
            network = add_neural_layer(network, in_features, out_features, activation)
            st.session_state.network = network

    if nn_model_col2.button('Reset Model'):
        st.session_state.network = None
        st.session_state.layer_n = None
        network = nn.Sequential()

    if nn_model_col3.button('Del Layer'):
        if len(network) > 0:
            network = network[:-1]
            st.session_state.network = network
            st.session_state.layer_n -= 1

    # visualise network
    st.text(summary(network))
    
else:
    model = None

st.markdown("## Model Fitting")
if st.sidebar.button("Train model"):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df_ml.drop(target, axis=1), 
                                                        df_ml[target], 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    
    #### THIS CAN UTILIZE THE CACHE FUNCTIONALITY
    # Train model and show progress
    if model_name == 'Neural Network':
        # convert data to tensors
        network = train_neural_network(network, X_train.values, y_train.values)
        model = network

    with st.spinner():
        model.fit(X_train, y_train)
    st.success('Model Fitted!')
    with st.spinner():
        y_pred = model.predict(X_test)
    st.success('Model tested!')
    if df_ml[target].dtype == 'object':
        st.write('Accuracy: ', accuracy_score(y_test, y_pred))
    else:
        st.write('Accuracy: ', model.score(X_test, y_test))
    # plot target
    ml_results_col1, ml_results_col2  = st.columns(2)

    # if the target is category, plot a bar chart of correct vs incorrect predictions within each category
    if df_ml[target].dtype == 'object':
        fig = px.histogram(y_pred == y_test, x=target, color=y_pred == y_test)
        ml_results_col1.plotly_chart(fig, use_container_width=True)
    else:
        # if the target is numeric, plot a histogram of residuals
        fig = px.histogram(abs(y_pred-y_test), x=target)
        ml_results_col1.plotly_chart(fig, use_container_width=True)

    # scatterplot of predicted and actual values
    fig = px.scatter(x=y_test, y=y_pred)
    ml_results_col2.plotly_chart(fig, use_container_width=True)








