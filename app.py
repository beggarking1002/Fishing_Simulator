from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

SECONDS_PER_DAY = 24 * 60 * 60

DATA_FILE_STEMS = {
    "restaurant": "Restaurant_Data",
    "fishing": "Fishing_Data",
    "guest": "Guest_Data",
}


@dataclass
class SimResult:
    catches_per_day: float
    fish_value_per_catch: float
    total_fish_value: float
    dish_count: float
    sales_capacity: float
    realized_sales: float
    gross_sales: float
    tips: float
    total_revenue: float
    bottleneck: str
    spawn_per_day: float
    avg_customer_cycle: float
    seat_capacity_per_day: float


# ---------- Loading helpers ----------

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Use the first row as header, drop the Korean description/type rows, and remove empty rows."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.iloc[0].tolist()]
    out = out.iloc[1:].reset_index(drop=True)
    out = out.dropna(how="all")

    # Remove rows that are likely description/type rows.
    kill_tokens = {
        "분류", "레벨", "비용", "등급", "이름", "Int", "Float", "Enum", "String",
        "string", "int", "float", "enum", "자료형", "계산 방식", "아이디", "텍스트 ID",
    }

    def row_is_meta(row: pd.Series) -> bool:
        vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
        if not vals:
            return True
        if all(v in kill_tokens for v in vals):
            return True
        return False

    out = out.loc[~out.apply(row_is_meta, axis=1)].reset_index(drop=True)
    return out


def find_data_file(base: Path, stem: str) -> Path:
    candidates = sorted(base.glob(f"{stem}*.xlsx")) + sorted(base.glob(f"{stem}*.xls")) + sorted(base.glob(f"{stem}*.ods"))
    if not candidates:
        raise FileNotFoundError(f"'{stem}'로 시작하는 xlsx/xls/ods 파일을 찾을 수 없습니다: {base}")
    return candidates[0]


def read_table(file_path: Path, sheet_name: str) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    engine = "odf" if suffix == ".ods" else None
    return pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, header=None)


@st.cache_data

def load_all_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)
    tables: Dict[str, pd.DataFrame] = {}

    restaurant_file = find_data_file(base, DATA_FILE_STEMS["restaurant"])
    fishing_file = find_data_file(base, DATA_FILE_STEMS["fishing"])
    guest_file = find_data_file(base, DATA_FILE_STEMS["guest"])

    tables["restaurant_upg"] = clean_table(read_table(restaurant_file, "Restaurant_UPG_Data"))
    tables["restaurant_settings"] = clean_table(read_table(restaurant_file, "Restaurant_UPG_Setting"))
    tables["recipes"] = clean_table(read_table(restaurant_file, "Recipe_Data"))

    tables["fishing_upg"] = clean_table(read_table(fishing_file, "Fishing_UPG_Data"))
    tables["fishing_settings"] = clean_table(read_table(fishing_file, "Fishing_UPG_Setting"))
    tables["rates"] = clean_table(read_table(fishing_file, "Fishing_Rate_Data"))
    tables["fish"] = clean_table(read_table(fishing_file, "Fish_Data"))

    tables["fixed"] = clean_table(read_table(guest_file, "Fixed_Value"))
    tables["guest_actions"] = clean_table(read_table(guest_file, "Customer_Action"))
    tables["guest_tips"] = clean_table(read_table(guest_file, "Customer_Tips"))

    for key, df in tables.items():
        tables[key] = df.replace({np.nan: None})

    return tables


# ---------- Data extraction ----------

def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_upgrade_level_lookup(upg_df: pd.DataFrame, max_only_settings: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    upg_df = to_numeric(upg_df, ["Level"])
    settings = to_numeric(max_only_settings, ["Max_Level"])
    max_from_data = upg_df.groupby("Upgrade_Type")["Level"].max().dropna().astype(int).to_dict()
    max_from_settings = settings.set_index("Upgrade_Type")["Max_Level"].dropna().astype(int).to_dict()
    all_keys = sorted(set(max_from_data) | set(max_from_settings))
    return {k: (1, int(max(max_from_data.get(k, 1), max_from_settings.get(k, 1)))) for k in all_keys}


def get_upgrade_value(df: pd.DataFrame, upgrade_type: str, level: int, value_col: str) -> float:
    dfn = to_numeric(df, ["Level", value_col])
    row = dfn[(dfn["Upgrade_Type"] == upgrade_type) & (dfn["Level"] == level)]
    if row.empty:
        row = dfn[dfn["Upgrade_Type"] == upgrade_type].sort_values("Level")
        if row.empty:
            return 0.0
        return float(row.iloc[-1][value_col])
    return float(row.iloc[0][value_col])


def get_rate_id_for_player_grade(df: pd.DataFrame, level: int) -> str:
    row = df[(df["Upgrade_Type"] == "PlayerGrade")].copy()
    row["Level"] = pd.to_numeric(row["Level"], errors="coerce")
    row = row[row["Level"] == level]
    if row.empty:
        all_rows = df[df["Upgrade_Type"] == "PlayerGrade"].copy()
        all_rows["Level"] = pd.to_numeric(all_rows["Level"], errors="coerce")
        all_rows = all_rows.sort_values("Level")
        return str(all_rows.iloc[-1]["Rate_ID"])
    return str(row.iloc[0]["Rate_ID"])


def expected_fish_value_per_catch(fish_df: pd.DataFrame, rate_df: pd.DataFrame, rate_id: str) -> float:
    fish = to_numeric(fish_df, ["Price"])
    rarity_prices = fish.groupby("Rarity")["Price"].mean().to_dict()
    rate_row = rate_df[rate_df["Gacha_Group_ID"] == rate_id]
    if rate_row.empty:
        return 0.0
    row = rate_row.iloc[0].to_dict()
    total = 0.0
    for rarity, avg_price in rarity_prices.items():
        pct = pd.to_numeric(row.get(rarity), errors="coerce")
        if pd.isna(pct):
            continue
        total += avg_price * (float(pct) / 100.0)
    return total


def recipe_value_per_fish(recipes_df: pd.DataFrame, fish_df: pd.DataFrame) -> float:
    recipes = to_numeric(recipes_df, ["Price", "Yield"])
    fish = to_numeric(fish_df, ["Price"])
    fish_price_map = fish.set_index("Fish_ID")["Price"].to_dict()
    vals = []
    for _, row in recipes.iterrows():
        ingredient = row.get("Ingredient")
        recipe_price = pd.to_numeric(row.get("Price"), errors="coerce")
        recipe_yield = pd.to_numeric(row.get("Yield"), errors="coerce")
        if pd.isna(recipe_price) or pd.isna(recipe_yield) or ingredient not in fish_price_map:
            continue
        fish_price = fish_price_map[ingredient]
        # Use the dish's total sales value generated from one fish ingredient.
        vals.append(float(recipe_price) * float(recipe_yield))
    return float(np.mean(vals)) if vals else 0.0


def weighted_guest_stats(actions_df: pd.DataFrame) -> dict:
    numeric_cols = [
        "Weight", "Flow_Velocity", "First_Order_Time", "First_Eat_Time", "Second_Order_Rate",
        "Second_Order_Time", "Second_Eat_Time", "Third_Order_Rate", "Third_Order_Time", "Third_Eat_Time",
    ]
    act = to_numeric(actions_df, numeric_cols)
    act = act.dropna(subset=["Weight"])
    weights = act["Weight"].astype(float)
    total_weight = weights.sum() if weights.sum() > 0 else 1.0

    def wavg(col: str) -> float:
        series = act[col].fillna(0).astype(float)
        return float((series * weights).sum() / total_weight)

    first_order = wavg("First_Order_Time")
    first_eat = wavg("First_Eat_Time")
    second_prob = wavg("Second_Order_Rate")
    second_order = wavg("Second_Order_Time")
    second_eat = wavg("Second_Eat_Time")
    third_prob = wavg("Third_Order_Rate")
    third_order = wavg("Third_Order_Time")
    third_eat = wavg("Third_Eat_Time")
    flow_velocity = wavg("Flow_Velocity")

    expected_orders = 1.0 + second_prob + second_prob * third_prob
    expected_cycle = (
        first_order + first_eat
        + second_prob * (second_order + second_eat)
        + second_prob * third_prob * (third_order + third_eat)
    )

    return {
        "expected_orders": expected_orders,
        "expected_cycle": expected_cycle,
        "avg_flow_velocity": flow_velocity,
    }


def weighted_tip_multiplier(actions_df: pd.DataFrame, tips_df: pd.DataFrame) -> float:
    act = to_numeric(actions_df, ["Weight"])
    act = act.dropna(subset=["Weight"])
    tips = to_numeric(tips_df, ["Tips_Rate", "Tips_Multi"])
    tip_map = tips.set_index("Grade")[["Tips_Rate", "Tips_Multi"]].to_dict("index")
    total_weight = act["Weight"].astype(float).sum()
    if total_weight <= 0:
        return 0.0
    acc = 0.0
    for _, row in act.iterrows():
        grade = row.get("Grade")
        weight = float(row.get("Weight") or 0)
        info = tip_map.get(grade, {"Tips_Rate": 0.0, "Tips_Multi": 0.0})
        acc += weight * float(info.get("Tips_Rate") or 0) * float(info.get("Tips_Multi") or 0)
    return acc / total_weight


# ---------- Simulation ----------

def run_simulation(tables: Dict[str, pd.DataFrame], levels: Dict[str, int], day_length: int) -> SimResult:
    fishing_upg = tables["fishing_upg"]
    rates = to_numeric(tables["rates"], ["Trash", "Normal", "Fine", "Superior", "Rare", "Elite", "Fantastic", "Legendary", "Sum"])
    fish = tables["fish"]
    recipes = tables["recipes"]
    fixed = to_numeric(tables["fixed"], ["DishWashTime", "MinSpawnDelay", "MaxSpawnDelay"])
    guest_actions = tables["guest_actions"]
    guest_tips = tables["guest_tips"]
    restaurant_upg = tables["restaurant_upg"]

    bait_interval = get_upgrade_value(fishing_upg, "BaitMaking", levels.get("BaitMaking", 1), "Effect_Value_Int")
    rod_count = get_upgrade_value(fishing_upg, "FishingRod", levels.get("FishingRod", 1), "Effect_Value_Int")
    ship_multiplier = get_upgrade_value(fishing_upg, "Ship", levels.get("Ship", 1), "Effect_Value_Int")
    if ship_multiplier <= 0:
        ship_multiplier = 1.0
    rate_id = get_rate_id_for_player_grade(fishing_upg, levels.get("PlayerGrade", 1))

    catches_per_rod = day_length / max(bait_interval, 1)
    catches_per_day = catches_per_rod * max(rod_count, 1) * ship_multiplier

    base_fish_value = expected_fish_value_per_catch(fish, rates, rate_id)
    dish_value_per_fish = recipe_value_per_fish(recipes, fish)
    effective_fish_to_dish_value = max(base_fish_value, dish_value_per_fish)
    dish_count = catches_per_day

    seat_count = get_upgrade_value(restaurant_upg, "Max_Customer_Limit", levels.get("Max_Customer_Limit", 1), "Effect_Value_Int")
    seat_count = max(seat_count, 1)
    spawn_bonus_special = get_upgrade_value(restaurant_upg, "Max_Spawn_Limit_1", levels.get("Max_Spawn_Limit_1", 1), "Effect_Value_Int")
    spawn_bonus_vip = get_upgrade_value(restaurant_upg, "Max_Spawn_Limit_2", levels.get("Max_Spawn_Limit_2", 1), "Effect_Value_Int")
    spawn_weight_bonus = get_upgrade_value(restaurant_upg, "Weight", levels.get("Weight", 1), "Effect_Value_Int")
    dish_price_bonus = (
        get_upgrade_value(restaurant_upg, "Bonus_Dish_Price_1", levels.get("Bonus_Dish_Price_1", 1), "Effect_Value_Int")
        + get_upgrade_value(restaurant_upg, "Bonus_Dish_Price_2", levels.get("Bonus_Dish_Price_2", 1), "Effect_Value_Int")
    )
    tip_bonus = get_upgrade_value(restaurant_upg, "Bonus_Tips_Multi", levels.get("Bonus_Tips_Multi", 1), "Effect_Value_Int")

    guest_stats = weighted_guest_stats(guest_actions)
    tip_multiplier = weighted_tip_multiplier(guest_actions, guest_tips)

    dish_wash_time = float(fixed.iloc[0]["DishWashTime"])
    min_spawn = float(fixed.iloc[0]["MinSpawnDelay"])
    max_spawn = float(fixed.iloc[0]["MaxSpawnDelay"])
    avg_spawn_gap = (min_spawn + max_spawn) / 2.0
    spawn_per_day = day_length / max(avg_spawn_gap, 1)
    spawn_per_day *= 1.0 + (spawn_bonus_special + spawn_bonus_vip + spawn_weight_bonus) * 0.05

    avg_customer_cycle = guest_stats["expected_cycle"] + dish_wash_time
    seat_capacity_per_day = seat_count * day_length / max(avg_customer_cycle, 1)
    customer_capacity = min(spawn_per_day, seat_capacity_per_day)

    expected_orders = guest_stats["expected_orders"]
    sales_capacity = customer_capacity * expected_orders
    realized_sales = min(dish_count, sales_capacity)

    avg_sale_price = effective_fish_to_dish_value + dish_price_bonus
    gross_sales = realized_sales * avg_sale_price
    tips = gross_sales * (tip_multiplier + tip_bonus / 100.0)
    total_revenue = gross_sales + tips

    if dish_count < sales_capacity:
        bottleneck = "낚시 공급 부족"
    elif sales_capacity < spawn_per_day:
        bottleneck = "좌석/체류시간 병목"
    else:
        bottleneck = "손님 유입 병목"

    return SimResult(
        catches_per_day=catches_per_day,
        fish_value_per_catch=base_fish_value,
        total_fish_value=catches_per_day * base_fish_value,
        dish_count=dish_count,
        sales_capacity=sales_capacity,
        realized_sales=realized_sales,
        gross_sales=gross_sales,
        tips=tips,
        total_revenue=total_revenue,
        bottleneck=bottleneck,
        spawn_per_day=spawn_per_day,
        avg_customer_cycle=avg_customer_cycle,
        seat_capacity_per_day=seat_capacity_per_day,
    )


# ---------- UI ----------

def main() -> None:
    st.set_page_config(page_title="Balance Simulator", layout="wide")
    st.title("밸런스 시뮬레이터 v0")
    st.caption("업로드한 XLSX/ODS 데이터 테이블을 읽어서 기대값 기반으로 낚시-식당 루프를 확인하는 Streamlit 프로토타입")

    with st.sidebar:
        st.header("데이터 경로")
        base_dir = st.text_input("데이터 파일 폴더", value=str(Path(__file__).resolve().parent))
        day_length = st.slider("시뮬레이션 길이(초)", min_value=1800, max_value=SECONDS_PER_DAY, value=SECONDS_PER_DAY, step=1800)

    try:
        tables = load_all_data(base_dir)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

    st.success("3개 데이터 테이블 로드 완료")

    fish_upgrade_ranges = build_upgrade_level_lookup(tables["fishing_upg"], tables["fishing_settings"])
    restaurant_upgrade_ranges = build_upgrade_level_lookup(tables["restaurant_upg"], tables["restaurant_settings"])

    st.subheader("업그레이드 레벨 설정")
    col1, col2 = st.columns(2)

    levels: Dict[str, int] = {}
    with col1:
        st.markdown("**Fishing**")
        for key, (min_lv, max_lv) in fish_upgrade_ranges.items():
            if key in {"분류", "Enum"}:
                continue
            levels[key] = st.slider(key, min_value=min_lv, max_value=max_lv, value=min_lv, key=f"fish_{key}")
    with col2:
        st.markdown("**Restaurant**")
        for key, (min_lv, max_lv) in restaurant_upgrade_ranges.items():
            if key in {"분류", "Enum", "Unlock_Gramophone", "Unlock_Cat_Object"}:
                continue
            levels[key] = st.slider(key, min_value=min_lv, max_value=max_lv, value=min_lv, key=f"rest_{key}")

    result = run_simulation(tables, levels, day_length)

    st.subheader("핵심 결과")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("일일 총수익", f"{result.total_revenue:,.0f}")
    k2.metric("총 판매량", f"{result.realized_sales:,.1f}")
    k3.metric("낚시 횟수", f"{result.catches_per_day:,.1f}")
    k4.metric("병목", result.bottleneck)

    st.subheader("세부 지표")
    detail_df = pd.DataFrame(
        {
            "Metric": [
                "Expected fish value / catch",
                "Total raw fish value",
                "Dish count available",
                "Sales capacity",
                "Spawn attempts / day",
                "Seat capacity / day",
                "Avg customer cycle (sec)",
                "Gross sales",
                "Tips",
                "Total revenue",
            ],
            "Value": [
                result.fish_value_per_catch,
                result.total_fish_value,
                result.dish_count,
                result.sales_capacity,
                result.spawn_per_day,
                result.seat_capacity_per_day,
                result.avg_customer_cycle,
                result.gross_sales,
                result.tips,
                result.total_revenue,
            ],
        }
    )
    st.dataframe(detail_df, use_container_width=True)

    st.subheader("병목 비교")
    compare_df = pd.DataFrame(
        {
            "Flow": ["Fishing supply", "Restaurant sales capacity"],
            "Value": [result.dish_count, result.sales_capacity],
        }
    )
    st.bar_chart(compare_df.set_index("Flow"))

    with st.expander("현재 모델 가정 보기"):
        st.markdown(
            """
- 현재 버전은 **기대값 기반 시뮬레이터**입니다. 즉, 손님/낚시 결과를 난수로 한 명씩 굴리지 않고 평균값으로 계산합니다.
- `Ship` 업그레이드는 현재 간단히 **낚시 횟수 배수**로 반영했습니다. 실제 의미가 다르면 수식만 교체하면 됩니다.
- 레시피는 `생선 1개 → 요리 Yield개 생산`으로 보고, 평균적인 요리 총매출 값을 계산합니다.
- 손님 로직은 `주문 시간 + 식사 시간 + 재주문 확률`의 기대값으로 좌석 점유 시간을 계산합니다.
- `Max_Spawn_Limit`, `Weight` 계열은 아직 실제 게임 수식이 없어서 **유입 보정치**로 단순 반영했습니다.
            """
        )

    with st.expander("원본 데이터 미리보기"):
        tab1, tab2, tab3 = st.tabs(["Fishing", "Restaurant", "Guest"])
        with tab1:
            st.dataframe(tables["fishing_upg"], use_container_width=True)
            st.dataframe(tables["rates"], use_container_width=True)
            st.dataframe(tables["fish"], use_container_width=True)
        with tab2:
            st.dataframe(tables["restaurant_upg"], use_container_width=True)
            st.dataframe(tables["recipes"], use_container_width=True)
        with tab3:
            st.dataframe(tables["fixed"], use_container_width=True)
            st.dataframe(tables["guest_actions"], use_container_width=True)
            st.dataframe(tables["guest_tips"], use_container_width=True)


if __name__ == "__main__":
    main()
