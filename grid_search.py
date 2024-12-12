from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier

def executar_grid_search(x_train_balanced, y_train_balanced):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=kfold,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(x_train_balanced, y_train_balanced)

    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor pontuação média na validação cruzada: {grid_search.best_score_:.4f}")

    # Exibe as melhores pontuações de validação e treino
    results = grid_search.cv_results_
    for mean_train_score, mean_test_score, params in zip(
        results['mean_train_score'], results['mean_test_score'], results['params']):
        print(f"Train Score: {mean_train_score:.4f}, Test Score: {mean_test_score:.4f} - Params: {params}")

    return grid_search.best_params_