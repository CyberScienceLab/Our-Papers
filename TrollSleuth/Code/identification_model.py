import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1].astype(int)
X, y = shuffle(X, y, random_state=0)

def classification_model(features):
    new_X = X[features]
    X_dict = new_X.to_dict(orient='records')
    vec = DictVectorizer()
    clf = XGBClassifier()
    pipeline = make_pipeline(vec, clf)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_pred = cross_val_predict(pipeline, X_dict, y, cv=kf)

    report = classification_report(y_pred, y, digits=4, output_dict=True)
    print(f'Classification Report:\n{report}')


def sensitivity_analysis():
    noise_levels = np.linspace(0.01, 0.1, 5)
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    category_impact_avg = {category: [] for category in feature_categories.keys()}
     
    for category, features in feature_categories.items():
        accuracy_drops = {noise_level: [] for noise_level in noise_levels}
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = xgboost.XGBClassifier()
            model.fit(X_train, y_train)
            y_pred_baseline = model.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
            for noise_level in noise_levels:
                perturbed_X_test = X_test.copy()
                noise = np.random.normal(0, noise_level, size=perturbed_X_test[features].shape)
                perturbed_X_test[features] += noise
                y_pred = model.predict(perturbed_X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_drop = (baseline_accuracy - accuracy) * 100 / baseline_accuracy
                accuracy_drops[noise_level].append(accuracy_drop)
        for noise_level in noise_levels:
            category_impact_avg[category].append(np.mean(accuracy_drops[noise_level]))
    
    heatmap_data = np.array([category_impact_avg[category] for category in feature_categories.keys()])
    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#dbe9f6", "#639cd9", "#004488"])
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 12},
        xticklabels=np.round(noise_levels, 2),
        yticklabels=list(feature_categories.keys()),
        cmap=custom_cmap,
        cbar_kws={'label': 'Accuracy Drop (%)'}
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.yaxis.label.set_size(14)
    colorbar.ax.tick_params(labelsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel('Noise Level', fontsize=16)
    plt.ylabel('Feature Category', fontsize=16)
    plt.tight_layout()
    plt.show()
      



