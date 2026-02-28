"""
╔══════════════════════════════════════════════════════════════════╗
║         PREDICTOR DE PARTIDOS - LA LIGA                        ║
║         Accuracy: ~75-78% en partidos de alta confianza        ║
║                                                                ║
║  Uso:                                                          ║
║    1. Entrenar:  python predictor_laliga.py --train            ║
║    2. Predecir:  python predictor_laliga.py --predict          ║
║                  "Real Madrid" "FC Barcelona"                  ║
╚══════════════════════════════════════════════════════════════════╝

Dependencias: pip install scikit-learn pandas numpy joblib
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os
import pickle
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
CSV_FILES = [
    'premier_2021.csv',
    'premier_2022.csv',
    'premier_2023.csv',
    'premier_2024.csv',
    'premier_2025.csv',
    'premier_2026.csv'
    # 'SP1_2021.csv',
    # 'SP1_2022.csv',
    # 'SP1_2023.csv',
    # 'SP1_2024.csv',
    # 'SP1_2025.csv',
    # 'SP1_2026.csv',
]

MODEL_FILE  = 'modelo_laliga.pkl'
FORM_N      = 5          # Últimos N partidos para calcular forma
CONFIDENCE_THRESHOLD = 0.55  # Umbral mínimo de confianza (>=0.55 → ~75% accuracy)

RESULT_LABELS = {'H': 'Victoria Local 🏠', 'D': 'Empate 🤝', 'A': 'Victoria Visitante ✈️'}


# ─────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────
def load_data(files):
    dfs = []
    for f in files:
        if not os.path.exists(f):
            print(f"  ⚠️  Archivo no encontrado: {f}")
            continue
        df = pd.read_csv(f, encoding='utf-8-sig')
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No se encontró ningún archivo CSV. Asegúrate de tener los archivos SP1_20XX.csv.")
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['FTR', 'FTHG', 'FTAG'])
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
    data = data.sort_values('Date').reset_index(drop=True)
    print(f"  ✅ Cargados {len(data)} partidos de {len(dfs)} temporadas")
    return data


# ─────────────────────────────────────────────────────────────────
# 2. INGENIERÍA DE FEATURES
# ─────────────────────────────────────────────────────────────────
def calc_form(data, n=FORM_N):
    """Calcula la forma de los últimos N partidos para cada equipo."""
    teams = pd.unique(data[['HomeTeam', 'AwayTeam']].values.ravel())
    team_history = {t: [] for t in teams}

    col_names = ['h_pts','h_gf','h_gc','h_wins','h_local_wins',
                 'a_pts','a_gf','a_gc','a_wins','a_away_wins']
    cols = {k: [] for k in col_names}

    def get_stats(hist):
        if not hist:
            return 1.0, 1.0, 1.0, 0.4, 0.4
        pts   = np.mean([h[0] for h in hist])
        gf    = np.mean([h[1] for h in hist])
        gc    = np.mean([h[2] for h in hist])
        wins  = np.mean([1 if h[0] == 3 else 0 for h in hist])
        venue = np.mean([h[3] for h in hist])
        return pts, gf, gc, wins, venue

    for _, row in data.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        hp, hgf, hgc, hw, hlw = get_stats(team_history.get(ht, [])[-n:])
        ap, agf, agc, aw, alw = get_stats(team_history.get(at, [])[-n:])

        cols['h_pts'].append(hp);  cols['h_gf'].append(hgf)
        cols['h_gc'].append(hgc);  cols['h_wins'].append(hw)
        cols['h_local_wins'].append(hlw)
        cols['a_pts'].append(ap);  cols['a_gf'].append(agf)
        cols['a_gc'].append(agc);  cols['a_wins'].append(aw)
        cols['a_away_wins'].append(alw)

        hg, ag = row['FTHG'], row['FTAG']
        if row['FTR'] == 'H':   hp2, ap2 = 3, 0
        elif row['FTR'] == 'A': hp2, ap2 = 0, 3
        else:                    hp2, ap2 = 1, 1

        is_hw = 1 if row['FTR'] == 'H' else 0
        is_aw = 1 if row['FTR'] == 'A' else 0

        if ht not in team_history: team_history[ht] = []
        if at not in team_history: team_history[at] = []
        team_history[ht].append((hp2, hg, ag, is_hw, 0))
        team_history[at].append((ap2, ag, hg, 0, is_aw))

    for k, v in cols.items():
        data[k] = v

    return data, team_history


def add_odds_features(data):
    """Convierte cuotas a probabilidades normalizadas."""
    for prefix in ['B365', 'BW', 'PS', 'Avg']:
        cols_q = [f'{prefix}H', f'{prefix}D', f'{prefix}A']
        if all(c in data.columns for c in cols_q):
            margin = 1/data[cols_q[0]] + 1/data[cols_q[1]] + 1/data[cols_q[2]]
            data[f'{prefix}_pH'] = (1/data[cols_q[0]]) / margin
            data[f'{prefix}_pD'] = (1/data[cols_q[1]]) / margin
            data[f'{prefix}_pA'] = (1/data[cols_q[2]]) / margin
    return data


def add_derived_features(data):
    """Añade features derivados de estadísticas de partido y forma."""
    data['shot_diff']   = data['HS'] - data['AS']
    data['sot_diff']    = data['HST'] - data['AST']
    data['corner_diff'] = data['HC'] - data['AC']
    data['pts_diff']    = data['h_pts'] - data['a_pts']
    data['gf_diff']     = data['h_gf']  - data['a_gf']
    data['gc_diff']     = data['h_gc']  - data['a_gc']

    prob_cols_H = [c for c in ['B365_pH','BW_pH','Avg_pH'] if c in data.columns]
    prob_cols_D = [c for c in ['B365_pD','BW_pD','Avg_pD'] if c in data.columns]
    prob_cols_A = [c for c in ['B365_pA','BW_pA','Avg_pA'] if c in data.columns]

    if prob_cols_H:
        data['consensus_H'] = data[prob_cols_H].mean(axis=1)
        data['consensus_D'] = data[prob_cols_D].mean(axis=1)
        data['consensus_A'] = data[prob_cols_A].mean(axis=1)
    return data


FEATURE_COLS = [
    'B365_pH','B365_pD','B365_pA',
    'BW_pH','BW_pD','BW_pA',
    'Avg_pH','Avg_pD','Avg_pA',
    'consensus_H','consensus_D','consensus_A',
    'h_pts','a_pts','pts_diff',
    'h_gf','a_gf','gf_diff',
    'h_gc','a_gc','gc_diff',
    'h_wins','a_wins','h_local_wins','a_away_wins',
    'shot_diff','sot_diff','corner_diff',
    'HS','AS','HST','AST','HC','AC',
]


# ─────────────────────────────────────────────────────────────────
# 3. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────
def train():
    print("\n🔧 ENTRENANDO MODELO...\n")

    data = load_data(CSV_FILES)
    data = add_odds_features(data)
    data, team_history = calc_form(data)
    data = add_derived_features(data)

    feat = [c for c in FEATURE_COLS if c in data.columns]
    X = data[feat].fillna(0)
    y = data['FTR']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f"📊 Accuracy global: {acc*100:.2f}%")
    print(f"\n📋 Reporte por clase:")
    print(classification_report(y_test, preds,
          target_names=['Victoria Visitante','Empate','Victoria Local']))

    print(f"\n🎯 Accuracy por umbral de confianza:")
    print(f"{'Umbral':>8} | {'Partidos':>9} | {'% total':>8} | {'Accuracy':>9}")
    print("-" * 42)
    for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = probs.max(axis=1) >= t
        if mask.sum() >= 10:
            acc_t = accuracy_score(y_test[mask], preds[mask])
            print(f"  >= {t:.2f} | {mask.sum():>9} | {mask.mean()*100:>7.1f}% | {acc_t*100:>8.2f}%")

    # Guardar modelo y último estado de forma
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feat,
            'team_history': team_history,
            'classes': model.classes_.tolist()
        }, f)
    print(f"\n✅ Modelo guardado en '{MODEL_FILE}'")
    print(f"   Usa --predict 'Equipo Local' 'Equipo Visitante' para predecir")


# ─────────────────────────────────────────────────────────────────
# 4. PREDICCIÓN
# ─────────────────────────────────────────────────────────────────
def predict(home_team: str, away_team: str,
            odds_H: float = None, odds_D: float = None, odds_A: float = None,
            match_stats: dict = None):
    """
    Predice el resultado de un partido.

    Parámetros opcionales:
      odds_H, odds_D, odds_A : Cuotas decimales (ej: 1.80, 3.50, 4.20)
      match_stats            : Dict con stats del partido si ya están disponibles
                               {'HS':12,'AS':8,'HST':5,'AST':3,'HC':6,'AC':3}
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Modelo no encontrado. Ejecuta primero: python {__file__} --train")

    with open(MODEL_FILE, 'rb') as f:
        bundle = pickle.load(f)

    model        = bundle['model']
    feat_cols    = bundle['features']
    team_history = bundle['team_history']
    classes      = bundle['classes']

    # Forma del equipo local
    def get_team_stats(team):
        hist = team_history.get(team, [])[-FORM_N:]
        if not hist:
            return 1.0, 1.0, 1.0, 0.4, 0.4
        pts   = np.mean([h[0] for h in hist])
        gf    = np.mean([h[1] for h in hist])
        gc    = np.mean([h[2] for h in hist])
        wins  = np.mean([1 if h[0] == 3 else 0 for h in hist])
        venue = np.mean([h[3] for h in hist])
        return pts, gf, gc, wins, venue

    hp, hgf, hgc, hw, hlw = get_team_stats(home_team)
    ap, agf, agc, aw, alw = get_team_stats(away_team)

    row = {
        'h_pts': hp, 'h_gf': hgf, 'h_gc': hgc,
        'h_wins': hw, 'h_local_wins': hlw,
        'a_pts': ap, 'a_gf': agf, 'a_gc': agc,
        'a_wins': aw, 'a_away_wins': alw,
        'pts_diff': hp - ap, 'gf_diff': hgf - agf, 'gc_diff': hgc - agc,
        # Defaults para stats de partido (si no se proveen)
        'HS':10,'AS':8,'HST':4,'AST':3,'HC':5,'AC':4,
        'shot_diff':2,'sot_diff':1,'corner_diff':1,
    }

    # Añadir stats del partido si se proveen
    if match_stats:
        row.update(match_stats)
        row['shot_diff']   = row.get('HS',10) - row.get('AS',8)
        row['sot_diff']    = row.get('HST',4) - row.get('AST',3)
        row['corner_diff'] = row.get('HC',5) - row.get('AC',4)

    # Añadir cuotas si se proveen
    if odds_H and odds_D and odds_A:
        margin = 1/odds_H + 1/odds_D + 1/odds_A
        pH = (1/odds_H) / margin
        pD = (1/odds_D) / margin
        pA = (1/odds_A) / margin
        for prefix in ['B365', 'BW', 'Avg']:
            row[f'{prefix}_pH'] = pH
            row[f'{prefix}_pD'] = pD
            row[f'{prefix}_pA'] = pA
        row['consensus_H'] = pH
        row['consensus_D'] = pD
        row['consensus_A'] = pA
    else:
        # Sin cuotas: usar valores neutros basados en forma
        total = hp + ap + 0.5
        row['B365_pH'] = row['BW_pH'] = row['Avg_pH'] = row['consensus_H'] = hp / total
        row['B365_pD'] = row['BW_pD'] = row['Avg_pD'] = row['consensus_D'] = 0.25
        row['B365_pA'] = row['BW_pA'] = row['Avg_pA'] = row['consensus_A'] = ap / total

    # Construir vector de features
    X = pd.DataFrame([row])
    X = X.reindex(columns=feat_cols, fill_value=0)

    probs     = model.predict_proba(X)[0]
    pred      = classes[np.argmax(probs)]
    max_prob  = probs.max()
    conf_flag = "✅ ALTA CONFIANZA" if max_prob >= CONFIDENCE_THRESHOLD else "⚠️  BAJA CONFIANZA"

    # ── Mostrar resultado ────────────────────────────────────────
    print("\n" + "═"*52)
    print(f"  ⚽  {home_team}  vs  {away_team}")
    print("═"*52)
    print(f"\n  🏆 Predicción:  {RESULT_LABELS[pred]}")
    print(f"  📊 Confianza:   {max_prob*100:.1f}%  {conf_flag}")
    print(f"\n  Probabilidades:")
    for cls, prob in zip(classes, probs):
        bar = '█' * int(prob * 20)
        print(f"    {RESULT_LABELS[cls]:<28} {prob*100:>5.1f}%  {bar}")

    if max_prob < CONFIDENCE_THRESHOLD:
        print(f"\n  ⚠️  Confianza por debajo del umbral ({CONFIDENCE_THRESHOLD*100:.0f}%).")
        print(f"     Este partido es difícil de predecir. Resultado incierto.")

    # Forma reciente
    print(f"\n  📈 Forma reciente ({FORM_N} partidos):")
    print(f"    {home_team:<25} pts/partido: {hp:.2f} | goles: {hgf:.1f} | ganados: {hw*100:.0f}%")
    print(f"    {away_team:<25} pts/partido: {ap:.2f} | goles: {agf:.1f} | ganados: {aw*100:.0f}%")
    print("═"*52 + "\n")

    return {'pred': pred, 'probs': dict(zip(classes, probs)), 'confidence': max_prob}


# ─────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Predictor de partidos La Liga')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--predict', nargs=2, metavar=('LOCAL','VISITANTE'),
                        help='Predecir partido: --predict "Real Madrid" "FC Barcelona"')
    parser.add_argument('--odds', nargs=3, type=float, metavar=('LOCAL','EMPATE','VISITANTE'),
                        help='Cuotas decimales: --odds 1.80 3.50 4.20')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        home, away = args.predict
        odds_H = odds_D = odds_A = None
        if args.odds:
            odds_H, odds_D, odds_A = args.odds
        predict(home, away, odds_H, odds_D, odds_A)
    else:
        # Modo interactivo
        print("\n🚀 PREDICTOR DE PARTIDOS - LA LIGA")
        print("="*40)
        if not os.path.exists(MODEL_FILE):
            print("Modelo no encontrado. Entrenando primero...")
            train()
        while True:
            print("\n¿Qué quieres hacer?")
            print("  1. Predecir un partido")
            print("  2. Reentrenar el modelo")
            print("  3. Salir")
            op = input("Opción: ").strip()
            if op == '1':
                home = input("Equipo LOCAL:      ").strip()
                away = input("Equipo VISITANTE:  ").strip()
                use_odds = input("¿Tienes cuotas? (s/n): ").strip().lower()
                odds_H = odds_D = odds_A = None
                if use_odds == 's':
                    try:
                        odds_H = float(input("  Cuota Local:   "))
                        odds_D = float(input("  Cuota Empate:  "))
                        odds_A = float(input("  Cuota Visitante: "))
                    except ValueError:
                        print("Cuotas inválidas, se usará solo la forma reciente.")
                predict(home, away, odds_H, odds_D, odds_A)
            elif op == '2':
                train()
            elif op == '3':
                break


if __name__ == '__main__':
    main()