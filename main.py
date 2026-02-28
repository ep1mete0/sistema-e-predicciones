"""
╔══════════════════════════════════════════════════════════════════╗
║         PREDICTOR DE PARTIDOS - LA LIGA                        ║
║         Accuracy: ~75-78% en partidos de alta confianza        ║
║                                                                ║
║  Comandos:                                                     ║
║    --train                      Entrenar el modelo             ║
║    --predict "Local" "Visit."   Predecir un partido            ║
║    --odds H D A                 Cuotas (junto a --predict)     ║
║    --resultado "L" "V" H|D|A   Registrar resultado real        ║
║    --goles H A                  Goles (junto a --resultado)    ║
║    --historial                  Ver partidos registrados       ║
║    --stats                      Ver estadísticas de aciertos   ║
║    (sin argumentos)             Modo interactivo               ║
╚══════════════════════════════════════════════════════════════════╝

Dependencias: pip install scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
CSV_FILES = [
    'premier/2021.csv',
    'premier/2022.csv',
    'premier/2023.csv',
    'premier/2024.csv',
    'premier/2025.csv',
    'premier/2026.csv'
]

MODEL_FILE     = 'modelo_laliga.pkl'
HISTORIAL_FILE = 'historial_partidos.json'
FORM_N         = 5
CONFIDENCE_THRESHOLD = 0.55

RESULT_LABELS = {'H': 'Victoria Local 🏠', 'D': 'Empate 🤝', 'A': 'Victoria Visitante ✈️'}


# ─────────────────────────────────────────────────────────────────
# HISTORIAL LOCAL (JSON)
# ─────────────────────────────────────────────────────────────────
def load_historial() -> list:
    if not os.path.exists(HISTORIAL_FILE):
        return []
    with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_historial(historial: list):
    with open(HISTORIAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)


def registrar_resultado(home: str, away: str, resultado: str,
                        goles_home: int = None, goles_away: int = None,
                        odds_H: float = None, odds_D: float = None, odds_A: float = None):
    """Guarda un partido jugado en el historial local para enriquecer futuros entrenamientos."""
    resultado = resultado.upper().strip()
    if resultado not in ('H', 'D', 'A'):
        print("❌ Resultado inválido. Usa H (local gana), D (empate) o A (visitante gana).")
        return

    historial = load_historial()
    hoy = datetime.now().strftime('%Y-%m-%d')

    # Detectar duplicado
    for p in historial:
        if p['HomeTeam'] == home and p['AwayTeam'] == away and p['Date'] == hoy:
            print(f"⚠️  Ya existe {home} vs {away} registrado hoy.")
            if input("¿Sobrescribir? (s/n): ").strip().lower() != 's':
                return
            historial.remove(p)
            break

    # Inferir goles si no se dan
    if goles_home is None or goles_away is None:
        if resultado == 'H':   goles_home, goles_away = goles_home or 1, goles_away or 0
        elif resultado == 'A': goles_home, goles_away = goles_home or 0, goles_away or 1
        else:                  goles_home, goles_away = goles_home or 1, goles_away or 1

    partido = {
        'Date': hoy, 'HomeTeam': home, 'AwayTeam': away,
        'FTR': resultado, 'FTHG': goles_home, 'FTAG': goles_away,
        # Stats de partido con valores por defecto neutros
        'HS': 10, 'AS': 8, 'HST': 4, 'AST': 3,
        'HC':  5, 'AC': 4, 'HY':  1, 'AY':  1,
        'HF': 10, 'AF': 10, 'HR': 0, 'AR':  0,
    }
    if odds_H and odds_D and odds_A:
        partido.update({
            'B365H': odds_H, 'B365D': odds_D, 'B365A': odds_A,
            'BWH':   odds_H, 'BWD':   odds_D, 'BWA':   odds_A,
            'AvgH':  odds_H, 'AvgD':  odds_D, 'AvgA':  odds_A,
        })

    historial.append(partido)
    save_historial(historial)

    print(f"\n✅ Partido registrado: {home} {resultado} {away}  ({goles_home}-{goles_away})")
    print(f"   Total en historial: {len(historial)} partido(s)")
    print(f"   Ejecuta --train para reentrenar el modelo con estos datos.\n")


def historial_to_df() -> pd.DataFrame:
    historial = load_historial()
    if not historial:
        return None
    df = pd.DataFrame(historial)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


# ─────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────
def load_data(files):
    dfs = []
    for f in files:
        if not os.path.exists(f):
            print(f"  ⚠️  No encontrado: {f}")
            continue
        dfs.append(pd.read_csv(f, encoding='utf-8-sig'))

    if not dfs:
        raise FileNotFoundError("No se encontró ningún CSV. Necesitas los archivos SP1_20XX.csv.")

    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['FTR', 'FTHG', 'FTAG'])
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

    # Incorporar historial manual
    extra = historial_to_df()
    if extra is not None and len(extra) > 0:
        data = pd.concat([data, extra], ignore_index=True)
        print(f"  ✅ Historial manual: +{len(extra)} partido(s) incorporado(s)")

    data = data.sort_values('Date').reset_index(drop=True)
    print(f"  ✅ Total partidos para entrenar: {len(data)}")
    return data


# ─────────────────────────────────────────────────────────────────
# INGENIERÍA DE FEATURES
# ─────────────────────────────────────────────────────────────────
def calc_form(data, n=FORM_N):
    teams = pd.unique(data[['HomeTeam', 'AwayTeam']].values.ravel())
    team_history = {t: [] for t in teams}
    cols = {k: [] for k in ['h_pts','h_gf','h_gc','h_wins','h_local_wins',
                             'a_pts','a_gf','a_gc','a_wins','a_away_wins']}

    def get_stats(hist):
        if not hist:
            return 1.0, 1.0, 1.0, 0.4, 0.4
        return (np.mean([h[0] for h in hist]),
                np.mean([h[1] for h in hist]),
                np.mean([h[2] for h in hist]),
                np.mean([1 if h[0] == 3 else 0 for h in hist]),
                np.mean([h[3] for h in hist]))

    for _, row in data.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        hp, hgf, hgc, hw, hlw = get_stats(team_history.get(ht, [])[-n:])
        ap, agf, agc, aw, alw = get_stats(team_history.get(at, [])[-n:])

        cols['h_pts'].append(hp);  cols['h_gf'].append(hgf);  cols['h_gc'].append(hgc)
        cols['h_wins'].append(hw); cols['h_local_wins'].append(hlw)
        cols['a_pts'].append(ap);  cols['a_gf'].append(agf);  cols['a_gc'].append(agc)
        cols['a_wins'].append(aw); cols['a_away_wins'].append(alw)

        hg = float(row.get('FTHG') or 1)
        ag = float(row.get('FTAG') or 0)
        hp2 = 3 if row['FTR']=='H' else (0 if row['FTR']=='A' else 1)
        ap2 = 3 if row['FTR']=='A' else (0 if row['FTR']=='H' else 1)
        is_hw = 1 if row['FTR']=='H' else 0
        is_aw = 1 if row['FTR']=='A' else 0

        if ht not in team_history: team_history[ht] = []
        if at not in team_history: team_history[at] = []
        team_history[ht].append((hp2, hg, ag, is_hw, 0))
        team_history[at].append((ap2, ag, hg, 0, is_aw))

    for k, v in cols.items():
        data[k] = v

    return data, team_history


def add_odds_features(data):
    for prefix in ['B365', 'BW', 'PS', 'Avg']:
        ch, cd, ca = f'{prefix}H', f'{prefix}D', f'{prefix}A'
        if all(c in data.columns for c in [ch, cd, ca]):
            margin = 1/data[ch] + 1/data[cd] + 1/data[ca]
            data[f'{prefix}_pH'] = (1/data[ch]) / margin
            data[f'{prefix}_pD'] = (1/data[cd]) / margin
            data[f'{prefix}_pA'] = (1/data[ca]) / margin
    return data


def add_derived_features(data):
    for col, default in [('HS',10),('AS',8),('HST',4),('AST',3),('HC',5),('AC',4)]:
        if col not in data.columns:
            data[col] = default

    data['shot_diff']   = data['HS']  - data['AS']
    data['sot_diff']    = data['HST'] - data['AST']
    data['corner_diff'] = data['HC']  - data['AC']
    data['pts_diff']    = data['h_pts'] - data['a_pts']
    data['gf_diff']     = data['h_gf']  - data['a_gf']
    data['gc_diff']     = data['h_gc']  - data['a_gc']

    ph = [c for c in ['B365_pH','BW_pH','Avg_pH'] if c in data.columns]
    pd_ = [c for c in ['B365_pD','BW_pD','Avg_pD'] if c in data.columns]
    pa = [c for c in ['B365_pA','BW_pA','Avg_pA'] if c in data.columns]
    if ph:
        data['consensus_H'] = data[ph].mean(axis=1)
        data['consensus_D'] = data[pd_].mean(axis=1)
        data['consensus_A'] = data[pa].mean(axis=1)
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
# ENTRENAMIENTO
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
        n_estimators=300, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42
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

    n_hist = len(load_historial())
    if n_hist:
        print(f"\n📝 Partidos del historial manual incluidos: {n_hist}")

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'features': feat,
                     'team_history': team_history, 'classes': model.classes_.tolist()}, f)

    print(f"\n✅ Modelo guardado en '{MODEL_FILE}'\n")


# ─────────────────────────────────────────────────────────────────
# PREDICCIÓN
# ─────────────────────────────────────────────────────────────────
def predict(home_team: str, away_team: str,
            odds_H: float = None, odds_D: float = None, odds_A: float = None):

    if not os.path.exists(MODEL_FILE):
        print("⚠️  Modelo no encontrado. Entrenando primero...\n")
        train()

    with open(MODEL_FILE, 'rb') as f:
        bundle = pickle.load(f)

    model        = bundle['model']
    feat_cols    = bundle['features']
    team_history = bundle['team_history']
    classes      = bundle['classes']

    def get_team_stats(team):
        hist = team_history.get(team, [])[-FORM_N:]
        if not hist:
            return 1.0, 1.0, 1.0, 0.4, 0.4
        return (np.mean([h[0] for h in hist]),
                np.mean([h[1] for h in hist]),
                np.mean([h[2] for h in hist]),
                np.mean([1 if h[0]==3 else 0 for h in hist]),
                np.mean([h[3] for h in hist]))

    hp, hgf, hgc, hw, hlw = get_team_stats(home_team)
    ap, agf, agc, aw, alw = get_team_stats(away_team)

    row = {
        'h_pts': hp, 'h_gf': hgf, 'h_gc': hgc, 'h_wins': hw, 'h_local_wins': hlw,
        'a_pts': ap, 'a_gf': agf, 'a_gc': agc, 'a_wins': aw, 'a_away_wins': alw,
        'pts_diff': hp-ap, 'gf_diff': hgf-agf, 'gc_diff': hgc-agc,
        'HS':10,'AS':8,'HST':4,'AST':3,'HC':5,'AC':4,
        'shot_diff':2,'sot_diff':1,'corner_diff':1,
    }

    if odds_H and odds_D and odds_A:
        margin = 1/odds_H + 1/odds_D + 1/odds_A
        pH, pD, pA = (1/odds_H)/margin, (1/odds_D)/margin, (1/odds_A)/margin
        for p in ['B365','BW','Avg']:
            row[f'{p}_pH'] = pH; row[f'{p}_pD'] = pD; row[f'{p}_pA'] = pA
        row['consensus_H'] = pH; row['consensus_D'] = pD; row['consensus_A'] = pA
    else:
        total = hp + ap + 0.5
        for p in ['B365','BW','Avg']:
            row[f'{p}_pH'] = hp/total; row[f'{p}_pD'] = 0.25; row[f'{p}_pA'] = ap/total
        row['consensus_H'] = hp/total; row['consensus_D'] = 0.25; row['consensus_A'] = ap/total

    X     = pd.DataFrame([row]).reindex(columns=feat_cols, fill_value=0)
    probs = model.predict_proba(X)[0]
    pred  = classes[np.argmax(probs)]
    conf  = probs.max()
    flag  = "✅ ALTA CONFIANZA" if conf >= CONFIDENCE_THRESHOLD else "⚠️  BAJA CONFIANZA"

    print("\n" + "═"*52)
    print(f"  ⚽  {home_team}  vs  {away_team}")
    print("═"*52)
    print(f"\n  🏆 Predicción:  {RESULT_LABELS[pred]}")
    print(f"  📊 Confianza:   {conf*100:.1f}%  {flag}")
    print(f"\n  Probabilidades:")
    for cls, prob in zip(classes, probs):
        bar = '█' * int(prob * 20)
        print(f"    {RESULT_LABELS[cls]:<28} {prob*100:>5.1f}%  {bar}")

    if conf < CONFIDENCE_THRESHOLD:
        print(f"\n  ⚠️  Confianza baja. Resultado incierto.")

    print(f"\n  📈 Forma reciente ({FORM_N} partidos):")
    print(f"    {home_team:<25} pts/p: {hp:.2f} | goles: {hgf:.1f} | ganados: {hw*100:.0f}%")
    print(f"    {away_team:<25} pts/p: {ap:.2f} | goles: {agf:.1f} | ganados: {aw*100:.0f}%")
    print("═"*52)
    print(f"\n  💡 Cuando se juegue, registra el resultado con:")
    print(f'     python predictor_laliga.py --resultado "{home_team}" "{away_team}" H|D|A\n')

    return {'pred': pred, 'probs': dict(zip(classes, probs)), 'confidence': conf}


# ─────────────────────────────────────────────────────────────────
# VER HISTORIAL
# ─────────────────────────────────────────────────────────────────
def ver_historial():
    historial = load_historial()
    if not historial:
        print("\n📭 El historial está vacío. Usa --resultado para registrar partidos.\n")
        return

    print(f"\n📋 HISTORIAL DE PARTIDOS ({len(historial)} total)")
    print("─"*62)
    print(f"  {'Fecha':<12} {'Local':<22} {'':>4}  {'Visitante':<22}")
    print("─"*62)
    for p in sorted(historial, key=lambda x: x['Date'], reverse=True):
        icon = {'H':'🏠','D':'🤝','A':'✈️'}.get(p['FTR'],'?')
        fthg = p.get('FTHG','?')
        ftag = p.get('FTAG','?')
        print(f"  {p['Date']:<12} {p['HomeTeam']:<22} {icon}{p['FTR']} {fthg}-{ftag}  {p['AwayTeam']}")
    print("─"*62 + "\n")


# ─────────────────────────────────────────────────────────────────
# ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────────
def ver_stats():
    historial = load_historial()
    if not historial:
        print("\n📭 No hay partidos registrados en el historial aún.\n")
        return

    total = len(historial)
    dist  = {'H': 0, 'D': 0, 'A': 0}
    for p in historial:
        dist[p['FTR']] = dist.get(p['FTR'], 0) + 1

    print(f"\n📊 ESTADÍSTICAS DEL HISTORIAL ({total} partidos)")
    print("─"*42)
    print(f"  🏠 Victorias locales:     {dist['H']:>3}  ({dist['H']/total*100:.1f}%)")
    print(f"  🤝 Empates:               {dist['D']:>3}  ({dist['D']/total*100:.1f}%)")
    print(f"  ✈️  Victorias visitante:   {dist['A']:>3}  ({dist['A']/total*100:.1f}%)")

    # Mini tabla por equipo
    equipos = {}
    for p in historial:
        for team, rol in [(p['HomeTeam'],'home'),(p['AwayTeam'],'away')]:
            if team not in equipos:
                equipos[team] = {'pts':0,'pg':0,'pe':0,'pp':0,'pj':0}
            equipos[team]['pj'] += 1
            if (rol=='home' and p['FTR']=='H') or (rol=='away' and p['FTR']=='A'):
                equipos[team]['pts'] += 3; equipos[team]['pg'] += 1
            elif p['FTR'] == 'D':
                equipos[team]['pts'] += 1; equipos[team]['pe'] += 1
            else:
                equipos[team]['pp'] += 1

    if len(equipos) >= 2:
        print(f"\n  🏆 Equipos en historial:")
        print(f"  {'Equipo':<25} {'PJ':>3} {'PG':>3} {'PE':>3} {'PP':>3} {'Pts':>4}")
        print("  " + "─"*40)
        for team, s in sorted(equipos.items(), key=lambda x: -x[1]['pts'])[:10]:
            print(f"  {team:<25} {s['pj']:>3} {s['pg']:>3} {s['pe']:>3} {s['pp']:>3} {s['pts']:>4}")

    print(f"\n  💡 Ejecuta --train para reentrenar con estos datos.\n")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Predictor La Liga con aprendizaje continuo')
    parser.add_argument('--train',      action='store_true')
    parser.add_argument('--predict',    nargs=2, metavar=('LOCAL','VISITANTE'))
    parser.add_argument('--odds',       nargs=3, type=float, metavar=('H','D','A'))
    parser.add_argument('--resultado',  nargs=3, metavar=('LOCAL','VISITANTE','RES'))
    parser.add_argument('--goles',      nargs=2, type=int, metavar=('H','A'))
    parser.add_argument('--historial',  action='store_true')
    parser.add_argument('--stats',      action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    elif args.resultado:
        home, away, res = args.resultado
        gh = args.goles[0] if args.goles else None
        ga = args.goles[1] if args.goles else None
        oH = args.odds[0] if args.odds else None
        oD = args.odds[1] if args.odds else None
        oA = args.odds[2] if args.odds else None
        registrar_resultado(home, away, res, gh, ga, oH, oD, oA)

    elif args.predict:
        home, away = args.predict
        oH = oD = oA = None
        if args.odds:
            oH, oD, oA = args.odds
        predict(home, away, oH, oD, oA)

    elif args.historial:
        ver_historial()

    elif args.stats:
        ver_stats()

    else:
        # ── MODO INTERACTIVO ─────────────────────────────────────
        print("\n╔══════════════════════════════════════╗")
        print("║   ⚽  PREDICTOR LA LIGA               ║")
        print("╚══════════════════════════════════════╝")

        if not os.path.exists(MODEL_FILE):
            print("\nModelo no encontrado. Entrenando primero...\n")
            train()

        while True:
            print("\n  1. Predecir un partido")
            print("  2. Registrar resultado real")
            print("  3. Ver historial")
            print("  4. Ver estadísticas")
            print("  5. Reentrenar modelo")
            print("  6. Salir")
            op = input("\nOpción: ").strip()

            if op == '1':
                home = input("Equipo LOCAL:       ").strip()
                away = input("Equipo VISITANTE:   ").strip()
                oH = oD = oA = None
                if input("¿Tienes cuotas? (s/n): ").strip().lower() == 's':
                    try:
                        oH = float(input("  Cuota Local:     "))
                        oD = float(input("  Cuota Empate:    "))
                        oA = float(input("  Cuota Visitante: "))
                    except ValueError:
                        print("Cuotas inválidas, continuando sin ellas.")
                predict(home, away, oH, oD, oA)

            elif op == '2':
                home = input("Equipo LOCAL:       ").strip()
                away = input("Equipo VISITANTE:   ").strip()
                print("  H = local gana  |  D = empate  |  A = visitante gana")
                res  = input("Resultado (H/D/A):  ").strip().upper()
                try:
                    gh_in = input("Goles LOCAL     (Enter para omitir): ").strip()
                    ga_in = input("Goles VISITANTE (Enter para omitir): ").strip()
                    gh = int(gh_in) if gh_in else None
                    ga = int(ga_in) if ga_in else None
                except ValueError:
                    gh = ga = None
                registrar_resultado(home, away, res, gh, ga)
                if input("¿Reentrenar el modelo ahora? (s/n): ").strip().lower() == 's':
                    train()

            elif op == '3':
                ver_historial()

            elif op == '4':
                ver_stats()

            elif op == '5':
                train()

            elif op == '6':
                print("\n¡Hasta luego! ⚽\n")
                break


if __name__ == '__main__':
    main()