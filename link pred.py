from py2neo import Graph
import pandas as pd
from numpy.random import randint
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import *
from pyspark.sql import functions as F
from sklearn.metrics import roc_curve, auc
from collections import Counter
from cycler import cycler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



graph = Graph("bolt://localhost", auth=("neo4j", "123"))
def down_sample(df):
    copy = df.copy()
    zero = Counter(copy.label.values)[0]
    un = Counter(copy.label.values)[1]
    n = zero - un
    copy = copy.drop(copy[copy.label == 0].sample(n=n, random_state=1).index)
    return copy.sample(frac=1)

# Find positive examples
train_existing_links = graph.run("""
MATCH (a:Character)-[:Interacts_early]->(b:Character)
RETURN id(a) AS node1, id(b) AS node2, 1 AS label
""").to_data_frame()# Find negative examples
train_missing_links = graph.run("""
MATCH (a:Character)
WHERE (a)-[:Interacts_early]-()
MATCH (a)-[:Interacts_early*2..3]-(other)
WHERE not((a)-[:Interacts_early]-(other))
RETURN id(a) AS node1, id(other) AS node2, 0 AS label
""").to_data_frame()# Remove duplicates
train_missing_links = train_missing_links.drop_duplicates()
training_df = train_missing_links.append(train_existing_links, ignore_index=True)
training_df['label'] = training_df['label'].astype('category')
training_df = down_sample(training_df)
training_data = spark.createDataFrame(training_df)

test_existing_links = graph.run("""
MATCH (a:Character)-[:Interacts_late]->(b:Character)
RETURN id(a) AS node1, id(b) AS node2, 1 AS label
""").to_data_frame()# Find negative examples
test_missing_links = graph.run("""
MATCH (a:Character)
WHERE (a)-[:Interacts_late]-()
MATCH (a)-[:Interacts_late*2..3]-(other)
WHERE not((a)-[:Interacts_late]-(other))
RETURN id(a) AS node1, id(other) AS node2, 0 AS label
""").to_data_frame()# Remove duplicates
test_missing_links = test_missing_links.drop_duplicates()
test_df = test_missing_links.append(test_existing_links, ignore_index=True)
test_df['label'] = test_df['label'].astype('category')
test_df = down_sample(test_df)
test_data = spark.createDataFrame(test_df)

test_data.groupby("label").count().show()



def create_pipeline(fields):
    assembler = VectorAssembler(inputCols=fields, outputCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                numTrees=30, maxDepth=10)
    return Pipeline(stages=[assembler, rf])




def apply_graphy_training_features(data):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
    pair.node2 AS node2,
    size([(p1)-[:Interacts_early]-(a)-
    [:Interacts_early]-(p2) | a]) AS Interactions,
    size((p1)-[:Interacts_early]-()) * size((p2)-
    [:Interacts_early]-()) AS prefAttachment,
    size(apoc.coll.toSet(
    [(p1)-[:Interacts_early]-(a) | id(a)] +
    [(p2)-[:Interacts_early]-(a) | id(a)]
    )) AS totalNeighbors
    """
    pairs = [{"node1": row["node1"], "node2": row["node2"]}
    for row in data.collect()]
    features = spark.createDataFrame(graph.run(query,
                                               {"pairs": pairs}).to_data_frame())
    return data.join(features, ["node1", "node2"])



def apply_graphy_test_features(data):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
    pair.node2 AS node2,
    size([(p1)-[:Interacts]-(a)-[:Interacts]-(p2) | a]) AS Interactions,
    size((p1)-[:Interacts]-()) * size((p2)-[:Interacts]-())
    AS prefAttachment,
    size(apoc.coll.toSet(
    [(p1)-[:Interacts]-(a) | id(a)] + [(p2)-[:Interacts]-(a) | id(a)]
    )) AS totalNeighbors
    """
    pairs = [{"node1": row["node1"], "node2": row["node2"]}
    for row in data.collect()]
    features = spark.createDataFrame(graph.run(query,
                                               {"pairs": pairs}).to_data_frame())
    return data.join(features, ["node1", "node2"])


training_data = apply_graphy_training_features(training_data)
test_data = apply_graphy_test_features(test_data)

plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
charts = [(1, "have interacted"), (0, "haven't interacted")]
for index, chart in enumerate(charts):
    label, title = chart
    filtered = training_data.filter(training_data["label"] == label)
    interactions = filtered.toPandas()["Interactions"]
    histogram =interactions.value_counts().sort_index()
    histogram /= float(histogram.sum())
    histogram.plot(kind="bar", x='Interactions', color="darkblue",
    ax=axs[index], title=f"Authors who {title} (label={label})")
    axs[index].xaxis.set_label_text("Interactions")
    
plt.tight_layout()
plt.show()


def train_model(fields, training_data):
    pipeline = create_pipeline(fields)
    model = pipeline.fit(training_data)
    return model

basic_model = train_model(["commonAuthors"], training_data)

eval_df = spark.createDataFrame(
        [(0,), (1,), (2,), (10,), (100,)],
        ['commonAuthors'])


(basic_model.transform(eval_df)
    .select("commonAuthors", "probability", "prediction")
    .show(truncate=False))


def evaluate_model(model, test_data):
    # Execute the model against the test set
    predictions = model.transform(test_data)
    # Compute true positive, false positive, false negative counts
    tp = predictions[(predictions.label == 1) &
        (predictions.prediction == 1)].count()
    fp = predictions[(predictions.label == 0) &
                     (predictions.prediction == 1)].count()
    fn = predictions[(predictions.label == 1) &
                     (predictions.prediction == 0)].count()
    # Compute recall and precision manually
    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)
    # Compute accuracy using Spark MLLib's binary classification evaluator
    accuracy = BinaryClassificationEvaluator().evaluate(predictions)
    # Compute false positive rate and true positive rate using sklearn functions
    labels = [row["label"] for row in predictions.select("label").collect()]
    preds = [row["probability"][1] for row in predictions.select
             ("probability").collect()]
    fpr, tpr, threshold = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return { "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "accuracy": accuracy,
            "recall": recall, "precision": precision }



def display_results(results):
    results = {k: v for k, v in results.items() if k not in ["fpr", "tpr", "roc_auc"]}
    return pd.DataFrame({"Measure": list(results.keys()),
                         "Score": list(results.values())})

basic_results = evaluate_model(basic_model, test_data)
display_results(basic_results)


def create_roc_plot():
    plt.style.use('classic')
    fig = plt.figure(figsize=(13, 8))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.rc('axes', prop_cycle=(cycler('color',
                                      ['r', 'g', 'b', 'c', 'm', 'y', 'k'])))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random score (AUC = 0.50)')
    return plt, fig

def add_curve(plt, title, fpr, tpr, roc):
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc:0.2})")


plt, fig = create_roc_plot()
add_curve(plt, "Common Authors",basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])
plt.legend(loc='lower right')
plt.show()

(training_data.filter(training_data["label"]==1)
    .describe()
    .select("summary", "commonAuthors", "prefAttachment", "totalNeighbors")
    .show())
(training_data.filter(training_data["label"]==0)
    .describe()
    .select("summary", "commonAuthors", "prefAttachment", "totalNeighbors")
    .show())


fields = ["commonAuthors", "prefAttachment", "totalNeighbors"]
graphy_model = train_model(fields, training_data)

graphy_results = evaluate_model(graphy_model, test_data)
display_results(graphy_results)



plt, fig = create_roc_plot()
add_curve(plt, "Common Authors",
          basic_results["fpr"], basic_results["tpr"],
                              basic_results["roc_auc"])
add_curve(plt, "Graphy",
          graphy_results["fpr"], graphy_results["tpr"],
          graphy_results["roc_auc"])
plt.legend(loc='lower right')
plt.show()


def plot_feature_importance(fields, feature_importances):
    df = pd.DataFrame({"Feature": fields, "Importance": feature_importances})
    df = df.sort_values("Importance", ascending=False)
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()


rf_model = graphy_model.stages[-1]
plot_feature_importance(fields, rf_model.featureImportances)


from spark_tree_plotting import export_graphviz
dot_string = export_graphviz(rf_model.trees[0],
                             featureNames=fields, categoryNames=[], classNames=["True", "False"],
                             filled=True, roundedCorners=True, roundLeaves=True)
with open("/tmp/rf.dot", "w") as file:
    file.write(dot_string)










