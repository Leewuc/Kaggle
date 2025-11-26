from config import np, pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {'speed', 'design_speed'}.issubset(df.columns):
        df['speed_violation_ratio'] = df['speed'] / (df['design_speed'] + 1e-3)
        df['is_speeding'] = (df['speed_violation_ratio'] > 1.0).astype(int)
        df['speed_squared'] = df['speed'] ** 2

        df['speed_bin'] = pd.cut(
            df['speed'],
            bins=[-0.001, 30, 60, 90, 200],
            labels=False,
            include_lowest=True
        ).fillna(0).astype(int) + 1
    else:
        df['speed_violation_ratio'] = 0.0
        df['is_speeding'] = 0
        df['speed_squared'] = 0.0
        df['speed_bin'] = 0

    if 'acceleration' in df.columns:
        df['hard_braking'] = (df['acceleration'] < -3).astype(int)
        df['rapid_acceleration'] = (df['acceleration'] > 3).astype(int)
        df['acc_abs'] = np.abs(df['acceleration'])
    else:
        df['hard_braking'] = 0
        df['rapid_acceleration'] = 0
        df['acc_abs'] = 0.0

    if {'acceleration', 'throttle_position'}.issubset(df.columns):
        df['aggressive_driving'] = df['acceleration'] * df['throttle_position']
    else:
        df['aggressive_driving'] = 0.0

    if {'rpm', 'engine_load_value'}.issubset(df.columns):
        df['engine_stress'] = df['rpm'] * df['engine_load_value'] / 100.0
        df['speed_rpm_ratio'] = df.get('speed', 0) / (df['rpm'] + 1.0) * 1000.0
        df['throttle_load_diff'] = df.get('throttle_position', 0) - df['engine_load_value']
    else:
        df['engine_stress'] = 0.0
        df['speed_rpm_ratio'] = 0.0
        df['throttle_load_diff'] = 0.0

    if 'rpm' in df.columns:
        df['engine_efficiency'] = df.get('speed', 0) / (df['rpm'] + 1.0)
        df['high_rpm'] = (df['rpm'] > 3000).astype(int)
    else:
        df['engine_efficiency'] = 0.0
        df['high_rpm'] = 0
    if {'current_weather', 'visibility', 'precipitation'}.issubset(df.columns):
        df['weather_risk'] = (
            df['current_weather'] *
            (1.0 / (df['visibility'] + 1.0)) *
            (df['precipitation'] + 1.0)
        )
        df['weather_visibility_ratio'] = df['current_weather'] / (df['visibility'] + 0.1)
        df['visibility_precipitation_interaction'] = df['visibility'] * df['precipitation']
        df['speed_weather_interaction'] = df.get('speed', 0) * df['current_weather']
        df['low_visibility'] = (df['visibility'] < 5).astype(int)
        df['heavy_precipitation'] = (df['precipitation'] > 10).astype(int)
    else:
        df['weather_risk'] = 0.0
        df['weather_visibility_ratio'] = 0.0
        df['visibility_precipitation_interaction'] = 0.0
        df['speed_weather_interaction'] = 0.0
        df['low_visibility'] = 0
        df['heavy_precipitation'] = 0

    if {'accidents_onsite', 'accidents_time'}.issubset(df.columns):
        df['location_risk'] = df['accidents_onsite'] + df['accidents_time']
        df['dangerous_location'] = (df['accidents_onsite'] > 50).astype(int)
    else:
        df['location_risk'] = 0.0
        df['dangerous_location'] = 0

    if 'observation_hour' in df.columns:
        df['is_rush_hour'] = df['observation_hour'].isin([7,8,9,17,18,19]).astype(int)
        df['is_night'] = ((df['observation_hour'] >= 20) | (df['observation_hour'] <= 5)).astype(int)
    else:
        df['is_rush_hour'] = 0
        df['is_night'] = 0

    if 'heart_rate' in df.columns:
        hr_mean, hr_std = df['heart_rate'].mean(), df['heart_rate'].std(ddof=0)
        df['heart_rate_zscore'] = (df['heart_rate'] - hr_mean) / (hr_std + 1e-6)
        df['heart_rate_bin'] = pd.cut(
            df['heart_rate'],
            bins=[0, 70, 90, 110, 200],
            labels=False,
            include_lowest=True
        ).fillna(0).astype(int) + 1
        df['driver_stress'] = (df['heart_rate_zscore'] > 1.0).astype(int)
    else:
        df['heart_rate_zscore'] = 0.0
        df['heart_rate_bin'] = 0
        df['driver_stress'] = 0

    if 'engine_temperature' in df.columns:
        df['engine_temp_normal'] = (
            (df['engine_temperature'] >= 80) &
            (df['engine_temperature'] <= 105)
        ).astype(int)
    else:
        df['engine_temp_normal'] = 1 

    df['total_risk_score'] = (
        df['speed_violation_ratio'] +
        df['weather_risk'] +
        df['location_risk']
    )
    df['speed_acceleration_product'] = df.get('speed', 0) * df['acc_abs']

    df['night_lowvis'] = (df['is_night'] & df['low_visibility']).astype(int)
    df['rush_lowvis'] = (df['is_rush_hour'] & df['low_visibility']).astype(int)
    df['speeding_lowvis'] = (df['is_speeding'] & df['low_visibility']).astype(int)

    df['speeding_heavy_precip'] = (df['is_speeding'] & df['heavy_precipitation']).astype(int)
    df['engine_stress_driver_stress'] = df['engine_stress'] * df['driver_stress']


    group_keys = []
    for col in ['driver_id', 'session_id', 'trip_id', 'road_segment_id']:
        if col in df.columns:
            group_keys.append(col)

    for gcol in group_keys:
        for base_col in ['speed', 'acceleration', 'heart_rate', 'weather_risk', 'engine_stress']:
            if base_col in df.columns:
                grp = df.groupby(gcol)[base_col]
                df[f'{gcol}_{base_col}_mean'] = df[gcol].map(grp.mean())
                df[f'{gcol}_{base_col}_std'] = df[gcol].map(grp.std())
                df[f'{base_col}_minus_{gcol}_mean'] = (
                    df[base_col] - df[f'{gcol}_{base_col}_mean']
                )

    return df