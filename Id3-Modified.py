import pandas as pd
from pprint import pprint
from sklearn.feature_selection import mutual_info_classif
from collections import Counter

# Function to build the ID3 Decision Tree using mutual information
def id3(df, target_attribute, attribute_names, default_class=None):
    cnt = Counter(x for x in df[target_attribute])  # Count occurrences of each class

    # Base cases
    if len(cnt) == 1:  # If all instances belong to one class
        return next(iter(cnt))
    elif df.empty or (not attribute_names):  # If no data or attributes remain
        return default_class

    # Recursive case
    else:
        # Calculate information gain for each attribute using mutual information
        gains = mutual_info_classif(df[attribute_names], df[target_attribute], discrete_features=True)
        index_of_max = gains.tolist().index(max(gains))  # Find the attribute with max information gain
        best_attr = attribute_names[index_of_max]

        # Initialize the tree with the best attribute as the root
        tree = {best_attr: {}}
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]

        # Recur for each subset of the best attribute
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute, remaining_attribute_names, default_class)
            tree[best_attr][attr_val] = subtree

        return tree

# Load the dataset
df = pd.read_csv("weather.csv")

# Remove the 'id' column as it is not a feature
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Prepare the dataset
attribute_names = df.columns.tolist()
attribute_names.remove("play")  # Remove target attribute

# Encode categorical variables
for colname in df.select_dtypes("object"):
    df[colname], _ = df[colname].factorize()

# Print the DataFrame
print("Dataset:")
print(df)

# Build the decision tree
tree = id3(df, "play", attribute_names)

# Print the resultant decision tree
print("The Resultant Decision Tree is:")
pprint(tree)

# Example classification function
def classify(instance, tree, default=None):
    attribute = next(iter(tree))  # Get the root attribute of the tree

    # Check if the instance's value exists in the tree
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]

        if isinstance(result, dict):  # If the result is another subtree, recurse
            return classify(instance, result)
        else:  # If the result is a class label, return it
            return result
    else:
        return default  # Default value if instance value is not in tree

# Test the decision tree with new instances
test_data = {
    'outlook': ['rainy', 'sunny'],
    'temperature': ['cool', 'hot'],
    'humidity': ['high', 'normal'],
    'wind': ['strong', 'weak']
}
df_test = pd.DataFrame(test_data)

# Classify instances and store predictions
df_test['Predicted'] = df_test.apply(classify, axis=1, args=(tree, 'No'))

# Print the test DataFrame with predictions
print("Test DataFrame with Predictions:")
print(df_test)