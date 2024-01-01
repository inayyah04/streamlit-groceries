import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

st.set_page_config(page_title="MBA Groceries", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)

with open("style.css") as f:
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown(
"""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #dc0000;">
  <a href="/" target="_self" id="main-btn">Market Basket Analysis</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav"">
      <li class="nav-item">
        <a id="notebook" class="nav-link active" href="https://www.kaggle.com/code/danielsimamora/market-basket-analysis" target="_blank">📄Notebook</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html = True)

st.markdown("""<p id="title-1z2x">Market Basket Analysis Groceries</p>""", unsafe_allow_html=True)
st.markdown("""<p id="caption-1z2x">Asosiasi item dengan menggunakan Apriori</p>""", unsafe_allow_html=True)

# Processing the CSV as Pandas DataFrame
df = pd.read_csv("Groceries_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

df["month"] = df['Date'].dt.month
df["day"] = df['Date'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace = True)
df["day"].replace([i for i in range(6 + 1)], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace = True)

# Filter the data based on User Inputs
def get_data(month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No result!"

st.sidebar.header('Nama : Innayah Azizah Latifah')
st.sidebar.subheader('NIM : 211351067')
st.sidebar.text("Silahkan masukkan nama item dan tanggal untuk dilakukan analisa")

def user_input_features():
    item = st.sidebar.selectbox("Item", ['tropical fruit', 'whole milk', 'pip fruit', 'other vegetables',
       'rolls/buns', 'citrus fruit', 'beef', 'frankfurter',
       'chicken', 'butter', 'fruit/vegetable juice',
       'packaged fruit/vegetables', 'chocolate', 'specialty bar',
       'butter milk', 'bottled water', 'yogurt', 'sausage', 'brown bread',
       'hamburger meat', 'root vegetables', 'pork', 'pastry',
       'canned beer', 'berries', 'coffee', 'misc. beverages', 'ham',
       'turkey', 'curd cheese', 'red/blush wine',
       'frozen potato products', 'flour', 'sugar', 'frozen meals',
       'herbs', 'soda', 'detergent', 'grapes', 'processed cheese', 'fish',
       'sparkling wine', 'newspapers', 'curd', 'pasta', 'popcorn',
       'finished products', 'beverages', 'bottled beer', 'dessert',
       'dog food', 'specialty chocolate', 'condensed milk', 'cleaner',
       'white wine', 'meat', 'ice cream', 'hard cheese', 'cream cheese',
       'liquor', 'pickled vegetables', 'liquor (appetizer)', 'uht-milk',
       'candy', 'onions', 'hair spray', 'photo/film', 'domestic eggs',
       'margarine', 'shopping bags', 'salt', 'oil', 'whipped/sour cream',
       'frozen vegetables', 'sliced cheese', 'dish cleaner',
       'baking powder', 'specialty cheese', 'salty snack',
       'instant food products', 'pet care', 'white bread',
       'female sanitary products', 'cling film/bags', 'soap',
       'frozen chicken', 'house keeping products', 'spread cheese',
       'decalcifier', 'frozen dessert', 'vinegar', 'nuts/prunes',
       'potato products', 'frozen fish', 'hygiene articles',
       'artif. sweetener', 'light bulbs', 'canned vegetables',
       'chewing gum', 'canned fish', 'cookware', 'semi-finished bread',
       'cat food', 'bathroom cleaner', 'prosecco', 'liver loaf',
       'zwieback', 'canned fruit', 'frozen fruits', 'brandy',
       'baby cosmetics', 'spices', 'napkins', 'waffles', 'sauces', 'rum',
       'chocolate marshmallow', 'long life bakery product', 'bags',
       'sweet spreads', 'soups', 'mustard', 'specialty fat',
       'instant coffee', 'snack products', 'organic sausage',
       'soft cheese', 'mayonnaise', 'dental care', 'roll products',
       'kitchen towels', 'flower soil/fertilizer', 'cereals',
       'meat spreads', 'dishes', 'male cosmetics', 'candles', 'whisky',
       'tidbits', 'cooking chocolate', 'seasonal products', 'liqueur',
       'abrasive cleaner', 'syrup', 'ketchup', 'cream', 'skin care',
       'rubbing alcohol', 'nut snack', 'cocoa drinks', 'softener',
       'organic products', 'cake bar', 'honey', 'jam', 'kitchen utensil',
       'flower (seeds)', 'rice', 'tea', 'salad dressing',
       'specialty vegetables', 'pudding powder', 'ready soups',
       'make up remover', 'toilet cleaner', 'preservation products'])
    month = st.sidebar.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.sidebar.select_slider('Day', ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value="Sat")

    return month, day, item

month, day, item = user_input_features()

data = get_data(month, day)

st.text("")
st.text("")
try:
  st.text("Dataset:")
  st.dataframe(data)
except:
  st.markdown("""<h4 style="text-align: center;">No transactions were done with that values 😕</h4>""", unsafe_allow_html=True)
  st.markdown("""
    <div id="ifno-result">
      <p>Here are some input values to give a try!</p>
      <ul style="margin: 0 auto">
        <li>Month: &nbsp;<i>Jan</i></li>
        <li>Day: &nbsp;<i>Sun</i></li>
      </ul>
    </div>
  """, unsafe_allow_html=True)


# ==========================================================================================================================================================================

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

if type(data) != type("No result!"):
  item_count = data.groupby(['Member_number','itemDescription'])["itemDescription"].count().reset_index(name = "Count")
  item_count_pivot = item_count.pivot_table(index='Member_number', columns='itemDescription', values='Count', aggfunc='sum').fillna(0)
  item_count_pivot = item_count_pivot.applymap(encode)

  support = 0.01 # atau 1%
  frequent_items = apriori(item_count_pivot, min_support = support, use_colnames = True)

  metric = "lift"
  min_threshold = 1

  rules = association_rules(frequent_items, metric = metric, min_threshold = min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
  rules.sort_values('confidence', ascending = False, inplace = True)

elif type(data) == type ("No result"):
    st.write("No Data")

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item):
    #data = rules[["antecedents", "consequents"]].copy()
    #data["antecedents"] = data["antecedents"].apply(parse_list)
    #data["consequents"] = data["consequents"].apply(parse_list)

    #return list(data.loc[data["antecedents"]==item_antecedents].iloc[0:])
    datax = rules[["antecedents", "consequents"]].copy()
    datax["antecedents"] = datax["antecedents"].apply(parse_list)
    datax["consequents"] = datax["consequents"].apply(parse_list)
    cobax = datax.loc[datax['antecedents'] == item].iloc[:,1].tolist()
    hasil = ", ".join(cobax)
    return hasil

    if hasil == "****":
      st.write("gue")




# ==========================================================================================================================================================================

st.text("")
st.text("")

if type(data) != type("No result!"):
  st.markdown("""<p id="recommendation-1z2x">Recommendation:</p>""", unsafe_allow_html=True)
  st.success(f"Customer who buys **{item}**, also buys **{return_item_df(item)}**!")