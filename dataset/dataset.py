#!/usr/bin/env python3
"""
HyGE-Mamba data preparation script v3.
============================================
Changes from v2:
  1. match_growth_curves() was replaced by match_growth_curves_multi()
     to extract four growth curves.
  2. HDF5 targets changed from scalar shape (N, 3) to curve shape (N, 4, T).
  3. The PyTorch DataLoader was adapted for four-curve output.
  4. VCF files are opened with encoding='utf-8' for Windows compatibility.

Other behavior remains compatible with v2.

Usage:
    python dataset.py -e example.csv -v new.vcf -g growth_curve.csv -o ./data
"""

import os, sys, argparse, math, time, warnings
import numpy as np
import pandas as pd
import h5py
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# Constants
GENOTYPE_ENCODING = {
    'AA': 0, 'AT': 1, 'TA': 1, 'AC': 2, 'CA': 2,
    'AG': 3, 'GA': 3, 'TT': 4, 'TC': 5, 'CT': 5,
    'TG': 6, 'GT': 6, 'CC': 7, 'CG': 8, 'GC': 8, 'GG': 9,
}
MISSING_ENCODING = -1
TBASE = 10.0

# Four growth-curve target names.
CURVE_NAMES = ['Maize.AboveGround.Wt', 'LAI', 'LeafWt', 'StemWt']

NASA_POWER_PARAMS = [
    'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR',
    'ALLSKY_SFC_PAR_TOT', 'CLRSKY_SFC_PAR_TOT',
    'ALLSKY_SFC_UVA', 'ALLSKY_SFC_UVB',
    'PS', 'WS2M', 'WD2M', 'GWETROOT', 'GWETPROF',
]

WEATHER_FEATURE_NAMES = [
    'Tmax', 'Tmin', 'DTR', 'TSR', 'MMR', 'GDD', 'dGDD',
    'RH', 'PR', 'PRDTR',
    'DayL', 'APAR', 'CPAR', 'UVA', 'UVB', 'PTT', 'dPTT', 'PTR', 'PTD1', 'PTD2',
    'PS', 'WS2M', 'WD', 'SW', 'SM'
]


# Part 1: data cleaning.
def clean_example_data(example_csv, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print(" Step 1: clean example data")
        print("="*60)
    df = pd.read_csv(example_csv)
    original_count = len(df)
    if verbose:
        print(f"  Raw rows: {original_count}")

    if 'plant_height_cm' in df.columns:
        n_miss = df['plant_height_cm'].isna().sum()
        df = df.dropna(subset=['plant_height_cm'])
        if verbose:
            print(f"\n  [plant_height_cm filter] missing: {n_miss}, remaining: {len(df)}")

    required = ['FEMALE', 'MALE', 'year_loc', 'date_plant', 'date_harvest',
                'yield', 'latitude', 'longitude']
    avail = [c for c in required if c in df.columns]
    if verbose:
        print(f"\n  Required-field missingness diagnostics ({len(df)} rows)")
        for col in avail:
            n_miss = df[col].isna().sum()
            flag = " <<<" if n_miss > 0 else ""
            print(f"    {col:<25s} : {n_miss:>6d} ({n_miss/max(len(df),1)*100:.1f}%){flag}")
    n_before = len(df)
    df = df.dropna(subset=avail)
    if verbose:
        print(f"\n  : -{n_before - len(df)} ")
        print(f"  : {len(df)}  ( {original_count - len(df)}/{original_count})")
    return df


def convert_date_format(date_str):
    if pd.isna(date_str) or date_str == '':
        return None
    for fmt in ['%Y/%m/%d', '%Y-%m-%d']:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).strftime('%Y-%m-%d')
        except:
            continue
    return None


def create_crosses_csv(df, output_path, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print(" 2:  crosses.csv")
        print("="*60)
    records = []
    for idx, row in df.iterrows():
        female = str(row.get('FEMALE', '')).replace('/', '_')
        male = str(row.get('MALE', '')).replace('/', '_')
        sid = f"{female}_{male}_{row.get('year_loc', 'unk')}_rep{row.get('rep', 1)}"
        records.append({
            'sample_id': sid,
            'parent_a': row['MALE'],
            'parent_b': row['FEMALE'],
            'genotype': row.get('genotype', ''),
            'location_id': row['year_loc'],
            'year': row.get('year', ''),
            'location': row.get('location', ''),
            'city': row.get('city', ''),
            'latitude': row.get('latitude', np.nan),
            'longitude': row.get('longitude', np.nan),
            'region': row.get('region', ''),
            'planting_date': convert_date_format(row['date_plant']),
            'harvest_date': convert_date_format(row['date_harvest']),
            'yield': row['yield'],
            'plant_height_cm': row.get('plant_height_cm', np.nan),
            'ear_height_cm': row.get('ear_height_cm', np.nan),
            'anthesis': row.get('anthesis', np.nan),
            'silking': row.get('silking', np.nan),
            'ASI': row.get('ASI', np.nan),
            'apsim_tt_emerg_to_endjuv': row.get('cult_tt_emerg_to_endjuv', np.nan),
            'apsim_tt_flower_to_start_grain': row.get('cult_tt_flower_to_start_grain', np.nan),
            'apsim_tt_flower_to_maturity': row.get('cult_tt_flower_to_maturity', np.nan),
            'apsim_photoperiod_slope': row.get('cult_photoperiod_slope', np.nan),
            'apsim_head_grain_no_max': row.get('cult_head_grain_no_max', np.nan),
            'apsim_grain_gth_rate': row.get('cult_grain_gth_rate', np.nan),
            'rep': row.get('rep', 1),
        })
    crosses_df = pd.DataFrame(records)
    crosses_df.to_csv(output_path, index=False)
    if verbose:
        print(f"  : {output_path}")
        print(f"  : {len(crosses_df)}, : {crosses_df['parent_a'].nunique()}, "
              f": {crosses_df['parent_b'].nunique()}, : {crosses_df['location_id'].nunique()}, "
              f": {crosses_df['genotype'].nunique()}")
    return crosses_df


# =============================================================================
# 2:   ()
# =============================================================================
def calculate_day_length(latitude, day_of_year):
    lat_rad = math.radians(latitude)
    dec = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    dec_rad = math.radians(dec)
    cos_ha = -math.tan(lat_rad) * math.tan(dec_rad)
    if cos_ha < -1: return 24.0
    elif cos_ha > 1: return 0.0
    return 2 * math.degrees(math.acos(cos_ha)) / 15


def fetch_nasa_power_data(lat, lon, start_date, end_date, max_retries=3):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        'parameters': ','.join(NASA_POWER_PARAMS),
        'community': 'AG', 'longitude': lon, 'latitude': lat,
        'start': start_date.replace('-', ''),
        'end': end_date.replace('-', ''), 'format': 'JSON'
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if 'properties' in data and 'parameter' in data['properties']:
                    df = pd.DataFrame(data['properties']['parameter'])
                    df.index = pd.to_datetime(df.index, format='%Y%m%d')
                    return df.replace(-999, np.nan)
            time.sleep(2 ** attempt)
        except:
            if attempt < max_retries - 1: time.sleep(2 ** attempt)
    return None


def compute_derived_weather(df, latitude):
    r = pd.DataFrame(index=df.index)
    Tmax = df.get('T2M_MAX', pd.Series(25, index=df.index))
    Tmin = df.get('T2M_MIN', pd.Series(15, index=df.index))
    r['Tmax'], r['Tmin'] = Tmax, Tmin
    r['DTR'] = Tmax - Tmin
    r['TSR'] = Tmax**2 - Tmin**2
    r['MMR'] = Tmin / Tmax.replace(0, np.nan)
    GDD = ((Tmax + Tmin) / 2 - TBASE).clip(lower=0)
    r['GDD'] = GDD
    r['dGDD'] = GDD.diff().abs().fillna(0)
    r['RH'] = df.get('RH2M', pd.Series(60, index=df.index))
    PR = df.get('PRECTOTCORR', pd.Series(0, index=df.index))
    r['PR'] = PR
    r['PRDTR'] = PR / r['DTR'].replace(0, np.nan)
    DayL = pd.Series(
        [calculate_day_length(latitude, d.timetuple().tm_yday) for d in df.index],
        index=df.index)
    r['DayL'] = DayL
    r['APAR'] = df.get('ALLSKY_SFC_PAR_TOT', pd.Series(30, index=df.index))
    r['CPAR'] = df.get('CLRSKY_SFC_PAR_TOT', pd.Series(35, index=df.index))
    r['UVA'] = df.get('ALLSKY_SFC_UVA', pd.Series(20, index=df.index))
    r['UVB'] = df.get('ALLSKY_SFC_UVB', pd.Series(1, index=df.index))
    PTT = GDD * DayL
    r['PTT'], r['dPTT'] = PTT, PTT.diff().abs().fillna(0)
    r['PTR'] = GDD / DayL.replace(0, np.nan)
    r['PTD1'] = r['DTR'] * DayL
    r['PTD2'] = r['DTR'] / DayL.replace(0, np.nan)
    r['PS'] = df.get('PS', pd.Series(101, index=df.index))
    r['WS2M'] = df.get('WS2M', pd.Series(2, index=df.index))
    r['WD'] = df.get('WD2M', pd.Series(180, index=df.index))
    r['SW'] = df.get('GWETROOT', pd.Series(0.5, index=df.index))
    r['SM'] = df.get('GWETPROF', pd.Series(0.5, index=df.index))
    return r.fillna(method='ffill').fillna(method='bfill').fillna(0)


def generate_synthetic_weather(start_date, end_date, latitude):
    dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq='D')
    np.random.seed(int(abs(latitude * 1000)) % 2**31)
    n = len(dates)
    data = {
        'T2M_MAX': 25+10*np.sin(np.linspace(0,2*np.pi,n))+np.random.randn(n)*2,
        'T2M_MIN': 15+8*np.sin(np.linspace(0,2*np.pi,n))+np.random.randn(n)*2,
        'RH2M': 60+np.random.randn(n)*10,
        'PRECTOTCORR': np.maximum(0, np.random.exponential(2,n)),
        'ALLSKY_SFC_PAR_TOT': 30+15*np.sin(np.linspace(0,np.pi,n)),
        'CLRSKY_SFC_PAR_TOT': 35+18*np.sin(np.linspace(0,np.pi,n)),
        'ALLSKY_SFC_UVA': 20+np.random.randn(n)*3,
        'ALLSKY_SFC_UVB': 1+np.random.randn(n)*0.2,
        'PS': 101+np.random.randn(n), 'WS2M': 2+np.random.exponential(1,n),
        'WD2M': np.random.uniform(0,360,n),
        'GWETROOT': 0.5+np.random.randn(n)*0.1, 'GWETPROF': 0.5+np.random.randn(n)*0.1,
    }
    df = pd.DataFrame(data, index=dates)
    df['T2M_MAX'] = df['T2M_MAX'].clip(10,45)
    df['T2M_MIN'] = df['T2M_MIN'].clip(0, df['T2M_MAX']-1)
    return compute_derived_weather(df, latitude)


def prepare_weather_data(crosses_df, max_days=180, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print(" 3:  (25)")
        print("="*60)
    ns = len(crosses_df)
    weather = np.zeros((ns, max_days, 25), dtype=np.float32)
    cache = {}
    for i, (_, row) in enumerate(crosses_df.iterrows()):
        lat, lon = row['latitude'], row['longitude']
        sd, ed = row['planting_date'], row['harvest_date']
        if pd.isna(lat) or pd.isna(lon) or pd.isna(sd) or pd.isna(ed):
            continue
        key = (lat, lon, sd, ed)
        if key not in cache:
            if verbose:
                print(f"   {i+1}/{ns}: ({lat:.2f},{lon:.2f}) {sd}~{ed}")
            raw = fetch_nasa_power_data(lat, lon, sd, ed)
            if raw is not None:
                cache[key] = compute_derived_weather(raw, lat)
            else:
                cache[key] = generate_synthetic_weather(sd, ed, lat)
        wf = cache[key]
        nd = min(len(wf), max_days)
        for fi, col in enumerate(WEATHER_FEATURE_NAMES):
            if col in wf.columns:
                weather[i, :nd, fi] = wf[col].values[:nd]
    if verbose:
        print(f"  : {weather.shape}, -: {len(cache)}")
    return weather


# =============================================================================
# 3: VCF  (,  open  encoding='utf-8')
# =============================================================================
def build_vcf_name_mapping(vcf_names, parent_names):
    """
    VCF  ->  , 3:
      1) : CG102 == CG102
      2) plink FID_IID : CG102_CG102 -> CG102
         : PHN11_LH145_0028_PHN11_LH145_0028 -> PHN11_LH145_0028
      3) : CG102_xxx -> CG102
    """
    mapping = {}
    parent_set = set(parent_names)

    #  plink : "B73_B73" -> "B73"
    plink_to_parent = {}
    for p in parent_names:
        plink_to_parent[f"{p}_{p}"] = p

    for vname in vcf_names:
        # 1: 
        if vname in parent_set:
            mapping[vname] = vname
            continue

        # 2: plink FID_IID 
        if vname in plink_to_parent:
            mapping[vname] = plink_to_parent[vname]
            continue

        # 3:  left==right
        found = False
        for sp in range(1, len(vname)):
            if vname[sp] == '_':
                left, right = vname[:sp], vname[sp+1:]
                if left == right and left in parent_set:
                    mapping[vname] = left
                    found = True
                    break
        if found:
            continue

        # 4:  _xxx
        parts = vname.rsplit('_', 1)
        if len(parts) == 2 and parts[0] in parent_set:
            mapping[vname] = parts[0]

    return mapping


def encode_genotype(ref, alt, gt):
    if gt in ['./.', '.|.', '.', '']:
        return MISSING_ENCODING
    sep = '/' if '/' in gt else '|'
    parts = gt.split(sep)
    try:
        a1, a2 = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return MISSING_ENCODING
    alleles = [ref] + alt.split(',')
    if a1 >= len(alleles) or a2 >= len(alleles):
        return MISSING_ENCODING
    pair = alleles[a1].upper() + alleles[a2].upper()
    return GENOTYPE_ENCODING.get(pair, MISSING_ENCODING)


def load_vcf_genotypes(vcf_path, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print(" 4: VCF (10)")
        print("="*60)
    geno_list, chrom_list, pos_list = [], [], []
    sample_names = []
    #  :  encoding='utf-8'  Windows GBK 
    with open(vcf_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line.startswith('##'): continue
            if line.startswith('#CHROM'):
                sample_names = line.split('\t')[9:]
                if verbose:
                    print(f"  VCF: {len(sample_names)}")
                    print(f"  5: {sample_names[:5]}")
                continue
            fields = line.split('\t')
            if len(fields) < 10: continue
            chrom, pos_val, ref, alt = fields[0], int(fields[1]), fields[3], fields[4]
            gts = fields[9:]
            row_g = []
            for i in range(len(sample_names)):
                if i < len(gts):
                    row_g.append(encode_genotype(ref, alt, gts[i].split(':')[0]))
                else:
                    row_g.append(MISSING_ENCODING)
            geno_list.append(row_g)
            c = chrom.replace('chr', '').replace('Chr', '')
            try: chrom_list.append(int(c) - 1)
            except: chrom_list.append(0)
            pos_list.append(pos_val)
    geno = np.array(geno_list).T  # (samples, snps)
    if verbose:
        print(f"  : {geno.shape}")
    return geno, np.array(chrom_list), np.array(pos_list), sample_names


def prepare_hybrid_genotypes(vcf_path, crosses_df, verbose=True):
    """, plink ()"""
    parents = sorted(set(crosses_df['parent_a'].unique()) |
                     set(crosses_df['parent_b'].unique()))
    geno, chroms, pos, vcf_names = load_vcf_genotypes(vcf_path, verbose)

    name_map = build_vcf_name_mapping(vcf_names, parents)
    parent_to_idx = {}
    for i, vn in enumerate(vcf_names):
        if vn in name_map:
            parent_to_idx[name_map[vn]] = i

    avail = set(parents) & set(parent_to_idx.keys())
    miss = set(parents) - avail
    if verbose:
        print(f"\n  : {len(avail)}/{len(parents)} ")
        if miss:
            print(f"  : {sorted(miss)[:10]}{'...' if len(miss)>10 else ''}")
        renamed = [(v, name_map[v]) for v in name_map if v != name_map[v]]
        if renamed:
            print(f"  plink:")
            for v, p in renamed[:5]:
                print(f"    {v} -> {p}")

    ga, gb = [], []
    for _, row in crosses_df.iterrows():
        pa, pb = row['parent_a'], row['parent_b']
        ga.append(geno[parent_to_idx[pa]] if pa in parent_to_idx
                  else np.full(geno.shape[1], MISSING_ENCODING, dtype=np.int8))
        gb.append(geno[parent_to_idx[pb]] if pb in parent_to_idx
                  else np.full(geno.shape[1], MISSING_ENCODING, dtype=np.int8))
    ga = np.array(ga, dtype=np.int8)
    gb = np.array(gb, dtype=np.int8)
    if verbose:
        print(f"  (MALE): {ga.shape}, (FEMALE): {gb.shape}")
    #  1D chroms/pos tile  
    return ga, gb, chroms, pos


# =============================================================================
# 4:  (  match_growth_curves)
# =============================================================================
def _double_logistic(t, L, k1, x01, k2, x02, b):
    """
    Double-logistic curve:
    y(t) = L * [sigmoid(k1*(t-x01)) - sigmoid(k2*(t-x02))] + b
    where sigmoid(x) = 1 / (1 + exp(-x)).

    Parameters:
        t: time points (np.ndarray)
        L: L_PotMax maximum potential
        k1: k1_Growth growth rate
        x01: x01_Inflection1 growth inflection point
        k2: k2_Senescence senescence rate
        x02: x02_Inflection2 senescence inflection point
        b: b_Offset baseline offset
    """
    # Numerically stable sigmoid.
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    y = L * (_sigmoid(k1 * (t - x01)) - _sigmoid(k2 * (t - x02))) + b
    return np.maximum(y, 0).astype(np.float32)


def match_growth_curves_multi(crosses_df, growth_curve_source, max_days=180,
                               curve_names=None, verbose=True):
    """
    Match four APSIM growth-curve time series.

    Supported input formats:
      A) Parameter format: FileName, Variable, L_PotMax, k1_Growth, ...
         Rebuild curves for t=0..max_days-1.
      B) Long format: Variable, SimulationValue, DayAfterSowing.
      C) Wide format: curve-name columns plus DayAfterSowing.

    Returns:
        curves: np.ndarray (N, 4, max_days) float32
    """
    if curve_names is None:
        curve_names = list(CURVE_NAMES)
    num_curves = len(curve_names)

    if verbose:
        print("\n" + "="*60)
        print(f" 5:  ({num_curves} )")
        print(f"  : {curve_names}")
        print("="*60)

    ns = len(crosses_df)
    curves = np.zeros((ns, num_curves, max_days), dtype=np.float32)

    source = str(growth_curve_source)
    if not os.path.exists(source):
        if verbose: print(f"   : {source}")
        return curves

    #   CSV 
    if os.path.isdir(source):
        csv_files = sorted(Path(source).glob('*.csv'))
        if verbose:
            print(f"  : {source}, CSV : {len(csv_files)}")
        all_dfs = []
        for fp in csv_files:
            try:
                all_dfs.append(pd.read_csv(fp))
            except:
                continue
        if all_dfs:
            gc = pd.concat(all_dfs, ignore_index=True)
        else:
            if verbose: print(f"    CSV")
            return curves
    else:
        gc = pd.read_csv(source)
        if verbose: print(f"  : {source}, : {len(gc)}")

    if verbose:
        print(f"  CSV : {list(gc.columns)}")

    PARAM_COLS = ['L_PotMax', 'k1_Growth', 'x01_Inflection1',
                  'k2_Senescence', 'x02_Inflection2', 'b_Offset']
    has_params = all(c in gc.columns for c in PARAM_COLS)
    has_variable = 'Variable' in gc.columns

    #  / 
    fn_col = None
    for c in ['FileName', 'filename', 'SimulationName', 'Name', 'Sample']:
        if c in gc.columns:
            fn_col = c
            break

    #  DAS 
    das_col = None
    for c in ['DayAfterSowing', 'DAS', 'DAP', 'Day', 'das']:
        if c in gc.columns:
            das_col = c
            break

    #  A:  ()
    #   FileName, Variable, L_PotMax, k1_Growth, x01, k2, x02, b_Offset
    if has_params and has_variable and fn_col:
        if verbose:
            print(f"  :  ( 6   )")
            print(f"  : {fn_col}, Variable : {gc['Variable'].unique().tolist()}")

        t = np.arange(max_days, dtype=np.float64)  # t = 0, 1, ..., max_days-1

        # : (yl, pp)  {variable: 6-param tuple}
        #  (yl, pp, rep)  ...
        param_idx = {}
        for _, row in gc.iterrows():
            fn = str(row[fn_col]).replace('.csv', '')
            var = str(row.get('Variable', ''))
            if var not in curve_names:
                continue
            try:
                parts = fn.split('__')
                if len(parts) >= 3:
                    yl, pp, rep = parts[0], parts[1], parts[2]
                elif len(parts) >= 2:
                    yl, pp = parts[0], parts[1]
                    rep = None
                else:
                    continue

                params = tuple(float(row[c]) if pd.notna(row[c]) else 0.0
                               for c in PARAM_COLS)

                #  rep  key
                if rep is not None:
                    param_idx[(yl, pp, rep)] = param_idx.get((yl, pp, rep), {})
                    param_idx[(yl, pp, rep)][var] = params
                #  rep  key ()
                if (yl, pp) not in param_idx:
                    param_idx[(yl, pp)] = {}
                param_idx[(yl, pp)][var] = params
            except:
                continue

        if verbose:
            print(f"   {len(param_idx)} ")
            var_counts = {}
            for key, vdict in param_idx.items():
                for v in vdict:
                    var_counts[v] = var_counts.get(v, 0) + 1
            for v, cnt in sorted(var_counts.items()):
                print(f"    {v:30s}: {cnt} ")

        #   crosses_df  
        matched = 0
        for i, (_, row) in enumerate(crosses_df.iterrows()):
            yl = row['location_id']
            f = str(row['parent_b']).replace('/', '_')  # FEMALE
            m = str(row['parent_a']).replace('/', '_')  # MALE
            rep = str(int(row.get('rep', 1)))

            #  key 
            found_params = None
            for key in [(yl, f"{f}_{m}", rep), (yl, f"{f}_{m}"),
                        (yl, f"{m}_{f}", rep), (yl, f"{m}_{f}")]:
                if key in param_idx:
                    found_params = param_idx[key]
                    break

            if found_params is None:
                continue

            matched += 1
            for ci, cn in enumerate(curve_names):
                if cn in found_params:
                    L, k1, x01, k2, x02, b = found_params[cn]
                    curves[i, ci, :] = _double_logistic(t, L, k1, x01, k2, x02, b)

        if verbose:
            print(f"\n  : {matched}/{ns}")
            for ci, cn in enumerate(curve_names):
                valid = (np.abs(curves[:, ci, :]).sum(axis=1) > 0).sum()
                if valid > 0:
                    vmax = curves[:, ci, :].max()
                    vmean = curves[curves[:, ci, :].sum(axis=1) > 0, ci, :].mean()
                    print(f"    {cn:30s}: {valid:5d} , max={vmax:.1f}, mean={vmean:.1f}")
                else:
                    print(f"    {cn:30s}: {valid:5d} ")

        return curves

    #  B:  (Variable + SimulationValue + DAS)
    if has_variable and 'SimulationValue' in gc.columns and das_col:
        if verbose: print(f"  :  (Variable + SimulationValue), DAS={das_col}")

        curve_data = {}
        for var_name in curve_names:
            sub = gc[gc['Variable'] == var_name]
            if verbose:
                print(f"    {var_name}: {len(sub)} ")
            for _, row in sub.iterrows():
                fn = str(row.get(fn_col, '') if fn_col else '')
                fn = fn.replace('.csv', '')
                if fn.endswith('_2'): fn = fn[:-2]
                das = int(row[das_col]) if pd.notna(row[das_col]) else -1
                val = float(row['SimulationValue']) if pd.notna(row['SimulationValue']) else 0.0

                try:
                    parts = fn.split('__')
                    if len(parts) >= 2:
                        yl, pp = parts[0], parts[1]
                        key = (yl, pp)
                        if key not in curve_data:
                            curve_data[key] = {cn: np.zeros(max_days, dtype=np.float32)
                                               for cn in curve_names}
                        if 0 <= das < max_days:
                            curve_data[key][var_name][das] = val
                except:
                    continue

        if verbose:
            print(f"   {len(curve_data)} ")

        matched = 0
        for i, (_, row) in enumerate(crosses_df.iterrows()):
            yl = row['location_id']
            f = str(row['parent_b']).replace('/', '_')
            m = str(row['parent_a']).replace('/', '_')
            for key in [(yl, f"{f}_{m}"), (yl, f"{m}_{f}")]:
                if key in curve_data:
                    for ci, cn in enumerate(curve_names):
                        curves[i, ci, :] = curve_data[key][cn]
                    matched += 1
                    break
        if verbose: print(f"  : {matched}/{ns}")
        return curves

    #  C:  ( + DAS)
    if all(cn in gc.columns for cn in curve_names) and das_col and fn_col:
        if verbose: print(f"  : ")

        curve_data = {}
        for _, row in gc.iterrows():
            fn = str(row[fn_col]).replace('.csv', '')
            if fn.endswith('_2'): fn = fn[:-2]
            das = int(row[das_col]) if pd.notna(row[das_col]) else -1
            try:
                parts = fn.split('__')
                if len(parts) >= 2:
                    yl, pp = parts[0], parts[1]
                    key = (yl, pp)
                    if key not in curve_data:
                        curve_data[key] = {cn: np.zeros(max_days, dtype=np.float32)
                                           for cn in curve_names}
                    if 0 <= das < max_days:
                        for cn in curve_names:
                            v = row.get(cn, 0)
                            curve_data[key][cn][das] = float(v) if pd.notna(v) else 0.0
            except:
                continue

        if verbose:
            print(f"   {len(curve_data)} ")

        matched = 0
        for i, (_, row) in enumerate(crosses_df.iterrows()):
            yl = row['location_id']
            f = str(row['parent_b']).replace('/', '_')
            m = str(row['parent_a']).replace('/', '_')
            for key in [(yl, f"{f}_{m}"), (yl, f"{m}_{f}")]:
                if key in curve_data:
                    for ci, cn in enumerate(curve_names):
                        curves[i, ci, :] = curve_data[key][cn]
                    matched += 1
                    break
        if verbose: print(f"  : {matched}/{ns}")
        return curves

    if verbose:
        print(f"   CSV ")
        print(f"     :")
        print(f"       : FileName, Variable, L_PotMax, k1_Growth, ...")
        print(f"       : Variable, SimulationValue, DayAfterSowing")
        print(f"       : {curve_names} + DayAfterSowing")
        print(f"     : {list(gc.columns)}")
    return curves


# =============================================================================
# 5:  M1-M5  ()
# =============================================================================
def _random_split_811(indices, seed_offset=0):
    """ 8:1:1"""
    n = len(indices)
    perm = np.random.RandomState(42 + seed_offset).permutation(indices)
    n_test = max(1, round(n * 0.1))
    n_val = max(1, round(n * 0.1))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return train_idx, val_idx, test_idx


def split_data_for_experiments(crosses_df, seed=42, verbose=True):
    """
    M1: ,  ( 8:1:1)
    M2: ,  ( 8:1:1)
    M3:  + 
    M4:  ( 8:1:1, )
    M5:  ( 8:1:1, )
    """
    if verbose:
        print("\n" + "="*60)
        print(" 6:  M1-M5 (train:val:test = 8:1:1)")
        print("="*60)
    np.random.seed(seed)
    n = len(crosses_df)
    idx = np.arange(n)
    envs = crosses_df['location_id'].unique()
    genos = crosses_df['genotype'].unique()
    splits = {}

    # ---- M1:  ----
    perm_g = np.random.permutation(genos)
    ng_t = max(1, round(len(genos) * 0.1))
    ng_v = max(1, round(len(genos) * 0.1))
    g_test  = set(perm_g[:ng_t])
    g_val   = set(perm_g[ng_t:ng_t + ng_v])
    g_train = set(perm_g[ng_t + ng_v:])
    splits['M1'] = {
        'train_indices': idx[crosses_df['genotype'].isin(g_train).values],
        'val_indices':   idx[crosses_df['genotype'].isin(g_val).values],
        'test_indices':  idx[crosses_df['genotype'].isin(g_test).values],
        'description': 'Known Env, Unknown Geno',
    }

    # ---- M2:  ----
    perm_e = np.random.permutation(envs)
    ne_t = max(1, round(len(envs) * 0.1))
    ne_v = max(1, round(len(envs) * 0.1))
    e_test  = set(perm_e[:ne_t])
    e_val   = set(perm_e[ne_t:ne_t + ne_v])
    e_train = set(perm_e[ne_t + ne_v:])
    splits['M2'] = {
        'train_indices': idx[crosses_df['location_id'].isin(e_train).values],
        'val_indices':   idx[crosses_df['location_id'].isin(e_val).values],
        'test_indices':  idx[crosses_df['location_id'].isin(e_test).values],
        'description': 'Unknown Env, Known Geno',
    }

    # ---- M3: + ----
    perm_e3 = np.random.permutation(envs)
    perm_g3 = np.random.permutation(genos)
    e3_test = set(perm_e3[:ne_t])
    g3_test = set(perm_g3[:ng_t])
    e3_val  = set(perm_e3[ne_t:ne_t + ne_v])
    g3_val  = set(perm_g3[ng_t:ng_t + ng_v])
    is_test = crosses_df['location_id'].isin(e3_test) | crosses_df['genotype'].isin(g3_test)
    is_val  = (~is_test) & (crosses_df['location_id'].isin(e3_val) | crosses_df['genotype'].isin(g3_val))
    is_train = (~is_test) & (~is_val)
    splits['M3'] = {
        'train_indices': idx[is_train.values],
        'val_indices':   idx[is_val.values],
        'test_indices':  idx[is_test.values],
        'description': 'Unknown Env + Unknown Geno',
    }

    # ---- M4:  ----
    tr4, va4, te4 = _random_split_811(idx, seed_offset=4)
    splits['M4'] = {
        'train_indices': tr4, 'val_indices': va4, 'test_indices': te4,
        'description': 'Env Only (no genotype)', 'use_genotype': False,
    }

    # ---- M5:  ----
    tr5, va5, te5 = _random_split_811(idx, seed_offset=5)
    splits['M5'] = {
        'train_indices': tr5, 'val_indices': va5, 'test_indices': te5,
        'description': 'Geno Only (no environment)', 'use_environment': False,
    }

    if verbose:
        for nm in ['M1', 'M2', 'M3', 'M4', 'M5']:
            s = splits[nm]
            tr = len(s['train_indices'])
            va = len(s['val_indices'])
            te = len(s['test_indices'])
            total = tr + va + te
            print(f"  {nm} ({s['description']}):")
            print(f"    ={tr}({tr/total*100:.0f}%), "
                  f"={va}({va/total*100:.0f}%), "
                  f"={te}({te/total*100:.0f}%)")
    return splits


# =============================================================================
# 6: HDF5  ( targets  (N,3)  (N,4,T))
# =============================================================================
def create_hdf5_dataset(out, ga, gb, chroms, pos, weather, crosses_df,
                        growth_curves=None, splits=None,
                        max_days=180, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print(" 7:  HDF5  ()")
        print("="*60)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ns = len(crosses_df)

    with h5py.File(out, 'w') as f:
        # --  () --
        f.create_dataset('genotypes_a', data=ga.astype(np.int8), compression='gzip')
        f.create_dataset('genotypes_b', data=gb.astype(np.int8), compression='gzip')

        # chromosomes / positions  1D 
        if chroms.ndim == 2:
            chroms = chroms[0]
        if pos.ndim == 2:
            pos = pos[0]
        f.create_dataset('chromosomes', data=chroms.astype(np.int8), compression='gzip')
        f.create_dataset('positions', data=pos.astype(np.int64), compression='gzip')
        f.attrs['is_shared_genomic_coords'] = True

        # --  () --
        f.create_dataset('weather_data', data=weather.astype(np.float32), compression='gzip')

        # --  : 4  (N, 4, T) --
        if growth_curves is not None:
            f.create_dataset('targets', data=growth_curves.astype(np.float32),
                             compression='gzip')
        else:
            f.create_dataset('targets',
                             data=np.zeros((ns, len(CURVE_NAMES), max_days), dtype=np.float32),
                             compression='gzip')

        # --  : yield, plant_height_cm, ear_height_cm (N, 3) --
        yield_arr   = crosses_df['yield'].values.astype(np.float32)
        height_arr  = crosses_df['plant_height_cm'].values.astype(np.float32)
        ear_arr     = crosses_df['ear_height_cm'].fillna(0).values.astype(np.float32)
        f.create_dataset('scalar_targets',
                         data=np.column_stack([yield_arr, height_arr, ear_arr]),
                         compression='gzip')
        f.attrs['scalar_target_names'] = 'yield,plant_height_cm,ear_height_cm'

        # -- APSIM () --
        ac = ['apsim_tt_emerg_to_endjuv', 'apsim_tt_flower_to_start_grain',
              'apsim_tt_flower_to_maturity', 'apsim_photoperiod_slope',
              'apsim_head_grain_no_max', 'apsim_grain_gth_rate']
        f.create_dataset('apsim_params', data=crosses_df[ac].fillna(0).values.astype(np.float32))

        # --  () --
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('sample_ids', data=crosses_df['sample_id'].values.astype(object), dtype=dt)
        f.create_dataset('location_ids', data=crosses_df['location_id'].values.astype(object), dtype=dt)
        f.create_dataset('genotype_ids', data=crosses_df['genotype'].values.astype(object), dtype=dt)

        # --  () --
        if splits:
            sg = f.create_group('splits')
            for nm, sd in splits.items():
                eg = sg.create_group(nm)
                eg.create_dataset('train_indices', data=sd['train_indices'])
                eg.create_dataset('val_indices', data=sd['val_indices'])
                eg.create_dataset('test_indices', data=sd['test_indices'])
                eg.attrs['description'] = sd['description']
                if 'use_genotype' in sd:
                    eg.attrs['use_genotype'] = sd['use_genotype']
                if 'use_environment' in sd:
                    eg.attrs['use_environment'] = sd['use_environment']

        # --   () --
        f.attrs['num_samples'] = ns
        f.attrs['num_snps'] = ga.shape[1]
        f.attrs['num_days'] = max_days
        f.attrs['num_weather_features'] = 25
        f.attrs['num_targets'] = len(CURVE_NAMES)
        f.attrs['num_timesteps'] = max_days
        f.attrs['target_names'] = ','.join(CURVE_NAMES)
        f.attrs['target_layout'] = 'NCT'   # (N, curves, timesteps)
        f.attrs['genotype_encoding'] = '10-class: AA(0),AT/TA(1),AC/CA(2),AG/GA(3),TT(4),TC/CT(5),TG/GT(6),CC(7),CG/GC(8),GG(9)'
        f.attrs['weather_feature_names'] = ','.join(WEATHER_FEATURE_NAMES)
        f.attrs['split_ratio'] = 'train:val:test = 8:1:1'

    sz = Path(out).stat().st_size / 1024 / 1024
    if verbose:
        print(f"  : {out} ({sz:.2f} MB)")
        print(f"  ={ns}, SNP={ga.shape[1]}, ={max_days}")
        print(f"  : {CURVE_NAMES}")
        print(f"  targets shape: ({ns}, {len(CURVE_NAMES)}, {max_days})")
    return out


# =============================================================================
# 7: PyTorch Dataset  ( )
# =============================================================================
LOADER_CODE = r'''"""
HyGE-Mamba HDF5 Dataset Loader ()
==========================================
:
    from hyge_dataset_loader import get_dataloaders

    train_ld, val_ld, test_ld = get_dataloaders(
        'data/hyge_mamba_dataset.h5', split='M1', batch_size=32
    )

    for batch in train_ld:
        geno_a  = batch['genotypes_a']     # (B, SNP)  long 0-9
        geno_b  = batch['genotypes_b']     # (B, SNP)  long
        chroms  = batch['chromosomes']     # (B, SNP)  long
        weather = batch['weather_data']    # (B, days, 25) float
        targets = batch['targets']         # (B, 4, T) float  4 
        apsim   = batch['apsim_params']    # (B, 6)
"""
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

CURVE_NAMES = ['Maize.AboveGround.Wt', 'LAI', 'LeafWt', 'StemWt']


class HyGEMambaDataset(Dataset):
    def __init__(self, h5_path, split='M1', partition='train'):
        self.h5_path = h5_path
        self._h5_file = None

        with h5py.File(h5_path, 'r') as f:
            self.indices = f[f'splits/{split}/{partition}_indices'][:]
            self.num_snps = int(f.attrs['num_snps'])
            self.num_days = int(f.attrs['num_days'])
            self.num_wf = int(f.attrs['num_weather_features'])
            sg = f[f'splits/{split}']
            self.use_geno = bool(sg.attrs.get('use_genotype', True))
            self.use_env = bool(sg.attrs.get('use_environment', True))
            self.has_apsim = 'apsim_params' in f

            is_shared = f.attrs.get('is_shared_genomic_coords', False)
            if is_shared or f['chromosomes'].ndim == 1:
                self._shared_chromosomes = torch.from_numpy(
                    f['chromosomes'][:].astype(np.int64))
                self._shared_positions = torch.from_numpy(
                    f['positions'][:].astype(np.int64))
            else:
                self._shared_chromosomes = torch.from_numpy(
                    f['chromosomes'][0].astype(np.int64))
                self._shared_positions = torch.from_numpy(
                    f['positions'][0].astype(np.int64))

    @property
    def h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5_file

    def __del__(self):
        if self._h5_file is not None:
            try: self._h5_file.close()
            except: pass

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ri = self.indices[idx]
        f = self.h5
        s = {}
        if self.use_geno:
            s['genotypes_a'] = torch.from_numpy(f['genotypes_a'][ri].astype(np.int64))
            s['genotypes_b'] = torch.from_numpy(f['genotypes_b'][ri].astype(np.int64))
            s['chromosomes'] = self._shared_chromosomes
        else:
            s['genotypes_a'] = torch.zeros(self.num_snps, dtype=torch.long)
            s['genotypes_b'] = torch.zeros(self.num_snps, dtype=torch.long)
            s['chromosomes'] = torch.zeros(self.num_snps, dtype=torch.long)
        s['positions'] = self._shared_positions
        if self.use_env:
            s['weather_data'] = torch.from_numpy(f['weather_data'][ri].astype(np.float32))
        else:
            s['weather_data'] = torch.zeros(self.num_days, self.num_wf)
        # targets: (4, T)  4 
        s['targets'] = torch.from_numpy(f['targets'][ri].astype(np.float32))
        if self.has_apsim:
            s['apsim_params'] = torch.from_numpy(f['apsim_params'][ri].astype(np.float32))
        else:
            s['apsim_params'] = torch.zeros(6)
        return s


def get_dataloaders(h5_path, split='M1', batch_size=32, num_workers=0):
    loaders = []
    for part in ['train', 'val', 'test']:
        ds = HyGEMambaDataset(h5_path, split, part)
        loaders.append(DataLoader(
            ds, batch_size=batch_size,
            shuffle=(part == 'train'),
            num_workers=num_workers,
            drop_last=(part == 'train'),
        ))
    return loaders[0], loaders[1], loaders[2]


def print_dataset_info(h5_path):
    with h5py.File(h5_path, 'r') as f:
        print(f"\n{'='*50}")
        print(f"HyGE-Mamba Dataset (): {h5_path}")
        print(f"{'='*50}")
        print(f":     {f.attrs['num_samples']}")
        print(f"SNP:      {f.attrs['num_snps']}")
        print(f":   {f.attrs['num_days']}")
        print(f":   {f.attrs['num_weather_features']}")
        print(f":   {f.attrs['target_names']}")
        tgt_shape = f['targets'].shape
        print(f"targets: {tgt_shape}")
        print(f":   {f.attrs['split_ratio']}")
        is_shared = f.attrs.get('is_shared_genomic_coords', False)
        print(f":   {'1D ' if is_shared else '2D '}")
        print(f"\n:")
        for k in sorted(f.keys()):
            if hasattr(f[k], 'shape'):
                print(f"  {k:20s} {str(f[k].shape):>20s}  {f[k].dtype}")
        print(f"\n:")
        for exp in ['M1', 'M2', 'M3', 'M4', 'M5']:
            if f'splits/{exp}' in f:
                sg = f[f'splits/{exp}']
                tr = len(sg['train_indices'])
                va = len(sg['val_indices'])
                te = len(sg['test_indices'])
                print(f"  {exp}: train={tr}, val={va}, test={te} | {sg.attrs['description']}")


if __name__ == '__main__':
    import sys
    h5_path = sys.argv[1] if len(sys.argv) > 1 else 'data/hyge_mamba_dataset.h5'
    print_dataset_info(h5_path)
'''


def create_dataset_loader(output_dir, verbose=True):
    loader_path = os.path.join(output_dir, 'hyge_dataset_loader.py')
    with open(loader_path, 'w') as fp:
        fp.write(LOADER_CODE)
    if verbose:
        print(f"\n  PyTorch DataLoader: {loader_path}")
    return loader_path


# =============================================================================
#   (  match_growth_curves_multi  match_growth_curves)
# =============================================================================
def prepare_dataset(example_csv, vcf_file, growth_curve_source, output_dir='./data',
                    max_days=180, seed=42, verbose=True):
    print("\n" + "="*60)
    print("  HyGE-Mamba  v3 ()")
    print("="*60)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = clean_example_data(example_csv, verbose)
    crosses_df = create_crosses_csv(df, os.path.join(output_dir, 'crosses.csv'), verbose)
    weather = prepare_weather_data(crosses_df, max_days, verbose)
    ga, gb, chroms, pos = prepare_hybrid_genotypes(vcf_file, crosses_df, verbose)

    #   ( match_growth_curves)
    growth_curves = match_growth_curves_multi(
        crosses_df, growth_curve_source, max_days=max_days, verbose=verbose)

    splits = split_data_for_experiments(crosses_df, seed, verbose)

    h5_path = os.path.join(output_dir, 'hyge_mamba_dataset.h5')
    create_hdf5_dataset(h5_path, ga, gb, chroms, pos, weather, crosses_df,
                        growth_curves=growth_curves, splits=splits,
                        max_days=max_days, verbose=verbose)
    loader_path = create_dataset_loader(output_dir, verbose)

    with open(os.path.join(output_dir, 'experiment_splits.txt'), 'w') as fp:
        fp.write("HyGE-Mamba Experiment Design (train:val:test = 8:1:1)\n")
        fp.write("=" * 55 + "\n\n")
        for nm, sd in splits.items():
            tr, va, te = len(sd['train_indices']), len(sd['val_indices']), len(sd['test_indices'])
            fp.write(f"{nm}: {sd['description']}\n  train={tr}, val={va}, test={te}\n\n")

    print(f"\n{'='*60}")
    print("  !")
    print(f"{'='*60}")
    print(f"\n: {h5_path}")
    print(f": {loader_path}")
    print(f"\n:")
    print(f"  from hyge_dataset_loader import get_dataloaders")
    print(f"  train_ld, val_ld, test_ld = get_dataloaders('{h5_path}', split='M1', batch_size=32)")
    print(f"  for batch in train_ld:")
    print(f"      targets = batch['targets']  # (B, 4, T) [AboveGround, LAI, LeafWt, StemWt]")
    return h5_path


def main():
    p = argparse.ArgumentParser(description='HyGE-Mamba data preparation v3')
    p.add_argument('--example_csv', '-e', required=True)
    p.add_argument('--vcf_file', '-v', required=True)
    p.add_argument('--growth_curve', '-g', required=True,
                   help='APSIM output CSV file or a directory containing CSV files')
    p.add_argument('--output_dir', '-o', default='./data')
    p.add_argument('--max_days', type=int, default=180)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quiet', '-q', action='store_true')
    a = p.parse_args()
    prepare_dataset(a.example_csv, a.vcf_file, a.growth_curve,
                    a.output_dir, a.max_days, a.seed, not a.quiet)


if __name__ == '__main__':
    main()
