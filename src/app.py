from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import math
import plotly.graph_objs as go
import math
import warnings


ASSESS_df=pd.read_csv('ASSESS_df.csv')
ASSESS_df=ASSESS_df[ASSESS_df['FY']>=2000]
RECC_df=pd.read_csv('RECC_df.csv')
ISADS=pd.read_csv('ISAD.csv')
options_df= None


# Dash app
app = dash.Dash(__name__)
server=app.server
# Layout of the app
app.layout = html.Div([
    html.H1("Frequency Table and Plot"),
    #Input box to find sic based on keywords if the user does not know what keyword to search
    dcc.Input(id='keywords', type='text', placeholder='Enter a keyword', style={'width': '80%', 'margin': '20px','height': '30px'}),
    html.Button('Submit', id='submit-button', n_clicks=0, style={'margin': '20px','height': '30px'}),
    dash_table.DataTable(id='output-table-1',
                        columns=[
                             {'name': 'SIC Code', 'id': 'SIC Code'},
                             {'name': 'Description', 'id': 'Description'},
                             {'name': 'Recommended', 'id': 'Recommended'},
                             {'name': 'Assessments', 'id': 'Assessments'},
                             {'name': 'Recommendations', 'id': 'Recommendations'},
                             {'name': 'Recommended $ Savings', 'id': 'Recommended $ Savings'}]
                        ),
    # Dropdown for SIC number
    dcc.Dropdown(sorted(ASSESS_df['SIC'].unique()), id='sic-input',style={'margin': '40px','height': '30px'},multi=True),

    # Output for plot 1
    dcc.Graph(id='output-plot-1',style={'margin': '40px'}),
    # Output for plot 2
    dcc.Graph(id='output-plot-2',style={'margin': '40px'}),
    html.H2("Assessment Table"),
    dash_table.DataTable(id='output-table-2',
                         columns=[
                            {'name': 'ID', 'id': 'ID'},
                            {'name': 'FY', 'id': 'FY'},
                            {'name': 'SIC', 'id': 'SIC'},
                            {'name': 'NAICS', 'id': 'NAICS'},
                            {'name': 'SALES', 'id': 'SALES'},
                            {'name': 'EMPLOYEES', 'id': 'EMPLOYEES'},
                            {'name': 'PRODUCTS', 'id': 'PRODUCTS'},
                            {'name': 'EC_plant_usage', 'id': 'EC_plant_usage'},
                            {'name': 'E2_plant_usage', 'id': 'E2_plant_usage'},
                         ],
                         sort_action='native',  # Enables native sorting
                         sort_mode='multi',  # Allows multi-column sorting
                        ),
    html.H2("Recommendation Table"),
    dash_table.DataTable(id='output-table-3',
                         columns=[
                             {'name': 'Description', 'id': 'Description'},
                             {'name': 'Recommended', 'id': 'Recommended'},
                             {'name': 'Count_I', 'id': 'Count_I'},
                             {'name': 'Count_N', 'id': 'Count_N'},
                             {'name': 'Count_P', 'id': 'Count_P'},
                             {'name': 'Imp_percent', 'id': 'Imp_percent'}
                         ],
                         sort_action='native',  # Enables native sorting
                         sort_mode='multi',  # Allows multi-column sorting
                        )
])
# Callback for 'sic-possibilities'
@app.callback(
    Output('output-table-1', 'data'),
    [Input('submit-button', 'n_clicks')],
    [State('keywords', 'value')]
)
def update_output_div(n_clicks, keyword):
    if n_clicks > 0:
        return update_output_for_keywords(keyword)
def update_output_for_keywords(input_keyword):
    
    # Read the CSV file
    df=pd.read_csv('sic_desc.csv')
    df=df[df['SIC Code']>1000]
    print(type(input_keyword))
    if input_keyword.isdigit():
        print(int(input_keyword))
        input_keyword=df[df['SIC Code']==int(input_keyword)]['Description']
        if not input_keyword.empty:
            # Convert the result to a string
            input_keyword = str(input_keyword.iloc[0])
            print(input_keyword)
        else:
            print("No matching description found for the given SIC Code.")
            return []
        
    print(input_keyword)
    if input_keyword is None:
        return []

    # Preprocess the data
    def preprocess_text(text):
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    # Apply preprocessing to the description column
    df['processed_description'] = df['Description'].apply(preprocess_text)
    # Function to find matching descriptions based on keywords
    def find_matching_descriptions(keyword, df):
        keyword = preprocess_text(keyword)
        matches = df[df['processed_description'].str.contains(keyword, case=False)]
        return matches
    matching_info = find_matching_descriptions(input_keyword, df)
    # Accessing SIC numbers from matching_info
    df_matches = df.loc[df['SIC Code'].isin(matching_info['SIC Code'])].copy()
    df_matches = df_matches.drop(['Link','processed_description'], axis=1)
    print(df_matches)
#     print("Matching SIC Numbers:")
#     print(df_matches)
    table_data = df_matches.to_dict('records')
    options_df=df_matches
    return table_data
# Callback to update frequency table, plots, and table based on input SIC
@app.callback(
    [Output('output-plot-1', 'figure'),
     Output('output-plot-2', 'figure'),
     Output('output-table-3', 'data'),
     Output('output-table-2', 'data')],
    [Input('sic-input', 'value')]
)
def update_output_for_sic(sic_input):
    print(sic_input)
    print(type(sic_input))
    if sic_input is None:
        return px.scatter(), px.scatter(), [], []  # Return default scatter plots if SIC is not valid

    # Filter DataFrame based on input SIC for Plot 1
    filtered_df_1 = ASSESS_df[ASSESS_df['SIC'].isin(sic_input)][['EC_plant_usage']].copy()
    filtered_df_1['EC_plant_usage'] = filtered_df_1['EC_plant_usage'].fillna(0).astype(int)
    z1=len(filtered_df_1[filtered_df_1['EC_plant_usage']==0])
    filtered_df_1=filtered_df_1[filtered_df_1['EC_plant_usage']!=0]
    # Calculate every 33% and get values as a NumPy array
    quartiles_1 = filtered_df_1['EC_plant_usage'].quantile([1/3, 2/3, 1]).values

    # Plot for Plot 1
    fig_1 = px.histogram(
        filtered_df_1,
        x='EC_plant_usage',
        title='Frequency of Yearly electricity consumption (kWh)',
        opacity=0.7,
        barmode='overlay',
        color_discrete_sequence=['black'],
        marginal="box",
        hover_data=[filtered_df_1['EC_plant_usage']],
        range_x=[0, math.ceil(max(filtered_df_1['EC_plant_usage'] / 1000000)) * 1000000],
        nbins=len(range(0, math.ceil(max(filtered_df_1['EC_plant_usage'] / 1000000)) * 1000000, 1000000))
    )

    fig_1.update_traces(marker_line_color='black', marker_line_width=1)  # Add black outline to bins
    # Create the frequency table using pd.cut()
    binsize=1000000
    frequency_table_1 = pd.cut(filtered_df_1['EC_plant_usage'], bins=range(0, filtered_df_1['EC_plant_usage'].max() + binsize, binsize))

    # Count the occurrences in each bin
    frequency_counts_1 = frequency_table_1.value_counts().sort_index()

    # Find the maximum count
    max_count_1 = frequency_counts_1.max()
    
    # Add shaded regions for every 33%
    fig_1.add_shape(
        type='rect',
        x0=0,
        x1=quartiles_1[0],
        y0=0,
        y1=1.3*max_count_1,
        fillcolor='rgba(39,43,84, 0.5)',
        line=dict(color='rgba(39,43,84, 0.5)', width=2)
    )

    fig_1.add_shape(
        type='rect',
        x0=quartiles_1[0],
        x1=quartiles_1[1],
        y0=0,
        y1=1.3*max_count_1,
        fillcolor='rgba(195,46,91, 0.5)',
        line=dict(color='rgba(195,46,91, 0.5)', width=2)
    )

    fig_1.add_shape(
        type='rect',
        x0=quartiles_1[1],
        x1=math.ceil(int(quartiles_1[2]) / 1000000) * 1000000,
        y0=0,
        y1=1.3*max_count_1,
        fillcolor='rgba(235,125,39, 0.5)',
        line=dict(color='rgba(235,125,39, 0.5)', width=2)
    )
    # Add hover labels for each 33% and 66% line with specified colors
    for q, label, color in zip(quartiles_1[:2], ['33%', '66%'], [(239,227,2), (126, 201, 76)]):
        fig_1.add_annotation(
            x=q,
            y=max_count_1,
            text=label,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color=f'rgb{color}')
        )
 
    # Filter DataFrame based on input SIC for Plot 2
    filtered_df_2 = ASSESS_df[ASSESS_df['SIC'].isin(sic_input)][['E2_plant_usage']].copy()
    filtered_df_2['E2_plant_usage'] = filtered_df_2['E2_plant_usage'].fillna(0).astype(int)
    z2=len(filtered_df_2[filtered_df_2['E2_plant_usage']==0])
    filtered_df_2=filtered_df_2[filtered_df_2['E2_plant_usage']!=0]
    # Calculate every 33% and get values as a NumPy array
    quartiles_2 = filtered_df_2['E2_plant_usage'].quantile([1/3, 2/3, 1]).values

    # Plot for Plot 2
    fig_2 = px.histogram(
        filtered_df_2,
        x='E2_plant_usage',
        title='Frequency of Yearly natural gas consumption (MMBtu)',
        opacity=0.7,
        barmode='overlay',
        color_discrete_sequence=['black'],
        marginal="box",
        hover_data=[filtered_df_2['E2_plant_usage']],
        range_x=[0, math.ceil(max(filtered_df_2['E2_plant_usage'] / 1000)) * 1000],
        nbins=len(range(0, math.ceil(max(filtered_df_2['E2_plant_usage'] / 1000)) * 1000, 1000))
    )
    fig_2.update_traces(marker_line_color='black', marker_line_width=1)  # Add black outline to bins
    # Create the frequency table using pd.cut()
    binsize=1000
    frequency_table_2 = pd.cut(filtered_df_2['E2_plant_usage'], bins=range(0, filtered_df_2['E2_plant_usage'].max() + binsize, binsize))

    # Count the occurrences in each bin
    frequency_counts_2 = frequency_table_2.value_counts().sort_index()

    # Find the maximum count
    max_count_2 = frequency_counts_2.max()

    # Add shaded regions for every 33%
    fig_2.add_shape(
        type='rect',
        x0=0,
        x1=int(quartiles_2[0]),
        y0=0,
        y1=1.3*max_count_2,
        fillcolor='rgba(39,43,84, 0.5)',
        line=dict(color='rgba(39,43,84, 0.5)', width=2)
    )

    fig_2.add_shape(
        type='rect',
        x0=int(quartiles_2[0]),
        x1=int(quartiles_2[1]),
        y0=0,
        y1=1.3*max_count_2,
        fillcolor='rgba(195,46,91, 0.5)',
        line=dict(color='rgba(195,46,91, 0.5)', width=2)
    )

    fig_2.add_shape(
        type='rect',
        x0=int(quartiles_2[1]),
        x1=int(math.ceil(max(filtered_df_2['E2_plant_usage'] / 1000)) * 1000),
        y0=0,
        y1=1.3*max_count_2,
        fillcolor='rgba(235,125,39, 0.5)',
        line=dict(color='rgba(235,125,39, 0.5)', width=2)
    )
    # Add hover labels for each 33% and 66% line with specified colors
    for q, label, color in zip(quartiles_2[:2], ['33%', '66%'], [(239,227,2), (126, 201, 76)]):
        fig_2.add_annotation(
            x=q,
            y=max_count_2,
            text=label,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color=f'rgb{color}')
        )
    
    filtered_df_3=ISADS[ISADS['SIC'].isin(sic_input)].copy()
    filtered_df_3['Count_N']=filtered_df_3['IMPSTATUS'].copy()
    filtered_df_3['Count_P']=filtered_df_3['IMPSTATUS'].copy()
    filtered_df_3 = filtered_df_3.groupby('ARC2').agg({
        'ID': 'count',
        'Description': 'first',
        'SIC': 'first',
        'IMPSTATUS': lambda x: (x == 'I').sum(),  # Count occurrences of 'I'
        'Count_N': lambda x: (x == 'N').sum(),  # Count occurrences of 'N'
        'Count_P': lambda x: (x == 'P').sum()  # Count occurrences of 'N'
    }).reset_index()
    filtered_df_3.rename(columns={'ID': 'Recommended','ARC2':'ARC', 'IMPSTATUS': 'Count_I'}, inplace=True)
    filtered_df_3['Imp_percent']=round(filtered_df_3['Count_I']*100/(filtered_df_3['Count_I']+filtered_df_3['Count_N']),2)
    tab=filtered_df_3[['ARC','Description','Recommended','Count_I','Count_N','Count_P','Imp_percent']].copy()
    tab.sort_values(by=['Imp_percent','Recommended'], ascending=False, inplace=True)
    table_data1 = tab.to_dict('records')

    filtered_df_4=ASSESS_df[ASSESS_df['SIC'].isin(sic_input)][['ID','FY','SIC','NAICS','SALES','EMPLOYEES','PRODUCTS','EC_plant_usage','E2_plant_usage']].copy()
    filtered_df_4['EC_plant_usage'] = filtered_df_4['EC_plant_usage'].fillna(0).astype(int)  
    filtered_df_4['E2_plant_usage'] = filtered_df_4['E2_plant_usage'].fillna(0).astype(int)
    filtered_df_4.sort_values(by=['SIC','FY'], ascending=True, inplace=True)
    table_data = filtered_df_4.to_dict('records')
    print(filtered_df_4)
    return fig_1, fig_2, table_data1, table_data
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
    print('Dash is running on http://127.0.0.1:8050/')
