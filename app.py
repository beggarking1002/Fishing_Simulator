from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import random

import numpy as np
import pandas as pd
import streamlit as st

SECONDS_PER_DAY = 24 * 60 * 60
HALF_TRIP_AT_V1 = 17.61 / 2.0
RARITY_COLS = ["Trash", "Normal", "Fine", "Superior", "Rare", "Elite", "Fantastic", "Legendary"]
DATA_FILE_STEMS = {
    "restaurant": "Restaurant_Data",
    "fishing": "Fishing_Data",
    "guest": "Guest_Data",
}
BASE_CUSTOMER_POOL = list(range(1, 15)) + [18, 19]
SPECIAL_EXTRA_ORDER = [15, 16, 17]
VIP_EXTRA_ORDER = [20]


@dataclass
class CustomerProfile:
    customer_id: int
    grade: str
    weight: float
    flow_velocity: float
    first_order_time: float
    first_eat_time: float
    second_order_rate: float
    second_order_time: float
    second_eat_time: float
    third_order_rate: float
    third_order_time: float
    third_eat_time: float
    tips_rate: float
    tips_multi: float


class SeatState:
    def __init__(self) -> None:
        self.phase = "ready_for_spawn"
        self.timer = 0.0
        self.customer: Optional[CustomerProfile] = None
        self.orders_completed = 0


@dataclass
class SimSummary:
    fish_caught: int
    fishing_sessions: int
    tickets_spent: int
    fish_left_in_inventory: int
    dishes_cooked: int
    dishes_sold: int
    gross_sales: float
    tips_sales: float
    total_sales: float
    customers_spawned: int
    customers_completed: int
    special_spawned: int
    vip_spawned: int
    peak_fish_inventory: int
    peak_dish_stock: int
    remaining_tickets: int
    remaining_dishes: int
    restaurant_idle_seconds: int
    bottleneck: str
    active_rate_id: str
    rarity_breakdown: pd.DataFrame
    fish_breakdown: pd.DataFrame
    recipe_breakdown: pd.DataFrame


# ---------------- data loading ----------------

def read_table(file_path: Path, sheet_name: str) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    engine = "odf" if suffix == ".ods" else None
    return pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, header=None)


def find_data_file(base: Path, stem: str) -> Path:
    candidates = sorted(base.glob(f"{stem}*.xlsx")) + sorted(base.glob(f"{stem}*.xls")) + sorted(base.glob(f"{stem}*.ods"))
    if not candidates:
        raise FileNotFoundError(f"{stem}로 시작하는 데이터 파일을 찾지 못했습니다: {base}")
    return candidates[0]


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.iloc[0].tolist()]
    out = out.iloc[1:].reset_index(drop=True)
    out = out.dropna(how="all")

    first_col_values_to_drop = {
        "분류", "아이디", "손님 ID", "등급", "이름", "설거지 소요 시간", "Upgrade_Type", "Recipe_ID", "Fish_ID",
    }
    row_tokens_to_drop = {
        "분류", "레벨", "비용", "등급", "이름", "Int", "Float", "Enum", "String", "string", "int", "float", "enum",
        "자료형", "계산 방식", "아이디", "텍스트 ID", "다음단계필요재화", "값", "참조아이디", "효과 값 1", "효과 값 2",
        "설거지 소요 시간", "최소 입장 딜레이", "최대 입장 딜레이", "한글 이름", "영어 이름", "가중치", "출입 속도",
        "첫 번째 주문 시간", "첫 번째 먹는 시간", "두 번째 주문 확률", "두 번째 주문 시간", "두 번째 먹는 시간",
        "세 번째 주문 확률", "세 번째 주문 시간", "세 번째 먹는 시간", "팁 확률", "팁 배수", "정가", "생산량", "재료",
    }

    def is_meta_row(row: pd.Series) -> bool:
        vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
        if not vals:
            return True
        if str(row.iloc[0]).strip() in first_col_values_to_drop:
            return True
        return all(v in row_tokens_to_drop for v in vals)

    out = out.loc[~out.apply(is_meta_row, axis=1)].reset_index(drop=True)
    return out.replace({np.nan: None})


@st.cache_data
def load_all_data(base_dir: str) -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)
    restaurant_file = find_data_file(base, DATA_FILE_STEMS["restaurant"])
    fishing_file = find_data_file(base, DATA_FILE_STEMS["fishing"])
    guest_file = find_data_file(base, DATA_FILE_STEMS["guest"])

    return {
        "restaurant_upg": clean_table(read_table(restaurant_file, "Restaurant_UPG_Data")),
        "restaurant_settings": clean_table(read_table(restaurant_file, "Restaurant_UPG_Setting")),
        "recipes": clean_table(read_table(restaurant_file, "Recipe_Data")),
        "fishing_upg": clean_table(read_table(fishing_file, "Fishing_UPG_Data")),
        "fishing_settings": clean_table(read_table(fishing_file, "Fishing_UPG_Setting")),
        "rates": clean_table(read_table(fishing_file, "Fishing_Rate_Data")),
        "fish": clean_table(read_table(fishing_file, "Fish_Data")),
        "fixed": clean_table(read_table(guest_file, "Fixed_Value")),
        "guest_actions": clean_table(read_table(guest_file, "Customer_Action")),
        "guest_tips": clean_table(read_table(guest_file, "Customer_Tips")),
    }


# ---------------- helpers ----------------

def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def get_upgrade_ranges(upg_df: pd.DataFrame, settings_df: pd.DataFrame) -> Dict[str, tuple[int, int]]:
    settings_df = to_numeric(settings_df, ["Max_Level"])
    max_from_settings = settings_df.set_index("Upgrade_Type")["Max_Level"].dropna().astype(int).to_dict()
    upg_df = to_numeric(upg_df, ["Level"])
    max_from_data = upg_df.groupby("Upgrade_Type")["Level"].max().dropna().astype(int).to_dict()
    keys = sorted(set(max_from_settings) | set(max_from_data))
    return {k: (1, int(max(max_from_settings.get(k, 1), max_from_data.get(k, 1)))) for k in keys}


def get_upgrade_value(upg_df: pd.DataFrame, upgrade_type: str, level: int, preferred: str) -> float:
    dfn = to_numeric(upg_df, ["Level", "Effect_Value_Int", "Effect_Value_Float"])
    row = dfn[(dfn["Upgrade_Type"] == upgrade_type) & (dfn["Level"] == level)]
    if row.empty:
        group = dfn[dfn["Upgrade_Type"] == upgrade_type].sort_values("Level")
        if group.empty:
            return 0.0
        row = group.tail(1)
    value = row.iloc[0].get(preferred)
    if pd.notna(value):
        return float(value)
    fallback = "Effect_Value_Float" if preferred == "Effect_Value_Int" else "Effect_Value_Int"
    value = row.iloc[0].get(fallback)
    if pd.notna(value):
        return float(value)
    return 0.0


def clamp_levels(levels: Dict[str, int]) -> Dict[str, int]:
    out = dict(levels)
    player_grade = int(out.get("PlayerGrade", 1))
    fishing_cap = max(1, (player_grade + 1) // 2)
    for key in ["BaitMaking", "FishingRod", "Ship"]:
        if key in out:
            out[key] = min(out[key], fishing_cap)

    master = int(out.get("Master_Lv", 1))
    for key in [
        "Max_Customer_Limit", "Max_Spawn_Limit_1", "Max_Spawn_Limit_2", "Weight", "Bonus_Tips_Multi",
        "Bonus_Dish_Price_1", "Bonus_Dish_Price_2", "Bonus_Food_1", "Bonus_Food_2",
    ]:
        if key in out:
            out[key] = min(out[key], master)
    return out


def get_rate_id_for_player_grade(fishing_upg_df: pd.DataFrame, player_grade: int) -> str:
    dfn = fishing_upg_df.copy()
    dfn["Level"] = pd.to_numeric(dfn["Level"], errors="coerce")
    row = dfn[(dfn["Upgrade_Type"] == "PlayerGrade") & (dfn["Level"] == player_grade)]
    if row.empty:
        row = dfn[dfn["Upgrade_Type"] == "PlayerGrade"].sort_values("Level").tail(1)
    return str(row.iloc[0]["Rate_ID"])


def normalize_rate(p: float) -> float:
    if pd.isna(p):
        return 0.0
    p = float(p)
    return p / 100.0 if p > 1.0 else p


def build_fishing_maps(tables: Dict[str, pd.DataFrame]):
    fish_df = to_numeric(tables["fish"], ["Price"])
    fish_df["Fish_ID"] = fish_df["Fish_ID"].astype(str)
    rates_df = to_numeric(tables["rates"], RARITY_COLS)

    rate_map: Dict[str, np.ndarray] = {}
    for _, row in rates_df.iterrows():
        rate_id = str(row["Gacha_Group_ID"])
        probs = np.array([float(row.get(col) or 0.0) for col in RARITY_COLS], dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(RARITY_COLS)) / len(RARITY_COLS)
        rate_map[rate_id] = probs

    rarity_to_fish: Dict[str, List[str]] = {}
    fish_rarity: Dict[str, str] = {}
    for _, row in fish_df.iterrows():
        fish_id = str(row["Fish_ID"])
        rarity = str(row["Rarity"])
        rarity_to_fish.setdefault(rarity, []).append(fish_id)
        fish_rarity[fish_id] = rarity
    return rate_map, rarity_to_fish, fish_rarity


def build_recipe_maps(recipes_df: pd.DataFrame, levels: Dict[str, int], restaurant_upg_df: pd.DataFrame):
    recipes_df = to_numeric(recipes_df, ["Price", "Yield"])
    bonus_food = (
        get_upgrade_value(restaurant_upg_df, "Bonus_Food_1", levels["Bonus_Food_1"], "Effect_Value_Int")
        + get_upgrade_value(restaurant_upg_df, "Bonus_Food_2", levels["Bonus_Food_2"], "Effect_Value_Int")
    )
    price_bonus = (
        get_upgrade_value(restaurant_upg_df, "Bonus_Dish_Price_1", levels["Bonus_Dish_Price_1"], "Effect_Value_Float")
        + get_upgrade_value(restaurant_upg_df, "Bonus_Dish_Price_2", levels["Bonus_Dish_Price_2"], "Effect_Value_Float")
    )

    fish_to_recipe: Dict[str, str] = {}
    recipe_price: Dict[str, float] = {}
    recipe_yield: Dict[str, int] = {}
    for _, row in recipes_df.iterrows():
        ingredient = str(row["Ingredient"])
        recipe_id = str(row["Recipe_ID"])
        base_price = float(row.get("Price") or 0.0)
        base_yield = int(float(row.get("Yield") or 0.0))
        fish_to_recipe[ingredient] = recipe_id
        recipe_price[recipe_id] = base_price * (1.0 + price_bonus)
        recipe_yield[recipe_id] = max(0, int(base_yield + bonus_food))
    return fish_to_recipe, recipe_price, recipe_yield


def build_customer_pool(tables: Dict[str, pd.DataFrame], levels: Dict[str, int]) -> List[CustomerProfile]:
    actions = to_numeric(
        tables["guest_actions"],
        [
            "Customer_ID", "Weight", "Flow_Velocity", "First_Order_Time", "First_Eat_Time", "Second_Order_Rate",
            "Second_Order_Time", "Second_Eat_Time", "Third_Order_Rate", "Third_Order_Time", "Third_Eat_Time",
        ],
    )
    tips = to_numeric(tables["guest_tips"], ["Tips_Rate", "Tips_Multi"])
    tips_by_grade = {}
    for _, row in tips.iterrows():
        tips_by_grade[str(row["Grade"])] = {
            "rate": normalize_rate(row.get("Tips_Rate") or 0.0),
            "multi": float(row.get("Tips_Multi") or 0.0),
        }

    special_effect = int(get_upgrade_value(tables["restaurant_upg"], "Max_Spawn_Limit_1", levels["Max_Spawn_Limit_1"], "Effect_Value_Int"))
    vip_effect = int(get_upgrade_value(tables["restaurant_upg"], "Max_Spawn_Limit_2", levels["Max_Spawn_Limit_2"], "Effect_Value_Int"))
    special_weight_bonus = float(get_upgrade_value(tables["restaurant_upg"], "Weight", levels["Weight"], "Effect_Value_Int"))
    tip_bonus_multi = float(get_upgrade_value(tables["restaurant_upg"], "Bonus_Tips_Multi", levels["Bonus_Tips_Multi"], "Effect_Value_Float"))

    allowed_ids = list(BASE_CUSTOMER_POOL)
    for i in range(min(special_effect, len(SPECIAL_EXTRA_ORDER))):
        allowed_ids.append(SPECIAL_EXTRA_ORDER[i])
    for i in range(min(vip_effect, len(VIP_EXTRA_ORDER))):
        allowed_ids.append(VIP_EXTRA_ORDER[i])
    allowed_ids = set(allowed_ids)

    pool: List[CustomerProfile] = []
    for _, row in actions.iterrows():
        cid = int(float(row["Customer_ID"]))
        if cid not in allowed_ids:
            continue
        grade = str(row["Grade"])
        weight = float(row.get("Weight") or 0.0)
        if grade == "Special":
            weight += special_weight_bonus
        tip_info = tips_by_grade.get(grade, {"rate": 0.0, "multi": 0.0})
        pool.append(
            CustomerProfile(
                customer_id=cid,
                grade=grade,
                weight=max(weight, 0.0001),
                flow_velocity=float(row.get("Flow_Velocity") or 1.0),
                first_order_time=float(row.get("First_Order_Time") or 0.0),
                first_eat_time=float(row.get("First_Eat_Time") or 0.0),
                second_order_rate=normalize_rate(row.get("Second_Order_Rate") or 0.0),
                second_order_time=float(row.get("Second_Order_Time") or 0.0),
                second_eat_time=float(row.get("Second_Eat_Time") or 0.0),
                third_order_rate=normalize_rate(row.get("Third_Order_Rate") or 0.0),
                third_order_time=float(row.get("Third_Order_Time") or 0.0),
                third_eat_time=float(row.get("Third_Eat_Time") or 0.0),
                tips_rate=tip_info["rate"],
                tips_multi=tip_info["multi"] * tip_bonus_multi,
            )
        )
    return pool


def pick_customer(pool: List[CustomerProfile], rng: random.Random) -> CustomerProfile:
    weights = [c.weight for c in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def pick_rarity_and_fish(rate_probs: np.ndarray, rarity_to_fish: Dict[str, List[str]], rng: random.Random):
    rarity = rng.choices(RARITY_COLS, weights=rate_probs, k=1)[0]
    fishes = rarity_to_fish.get(rarity, [])
    if not fishes:
        all_fishes = [f for arr in rarity_to_fish.values() for f in arr]
        fish_id = rng.choice(all_fishes)
        return rarity, fish_id
    return rarity, rng.choice(fishes)


def random_delay(fixed_df: pd.DataFrame, rng: random.Random) -> float:
    row = to_numeric(fixed_df, ["DishWashTime", "MinSpawnDelay", "MaxSpawnDelay"]).iloc[0]
    return rng.uniform(float(row["MinSpawnDelay"]), float(row["MaxSpawnDelay"]))


def run_simulation(tables: Dict[str, pd.DataFrame], levels: Dict[str, int], total_seconds: int, wait_after_full_seconds: int, seed: int) -> SimSummary:
    rng = random.Random(seed)
    levels = clamp_levels(levels)

    fishing_upg = tables["fishing_upg"]
    restaurant_upg = tables["restaurant_upg"]
    fixed_df = to_numeric(tables["fixed"], ["DishWashTime", "MinSpawnDelay", "MaxSpawnDelay"])
    fixed_row = fixed_df.iloc[0]

    bait_seconds = int(get_upgrade_value(fishing_upg, "BaitMaking", levels["BaitMaking"], "Effect_Value_Int"))
    rod_capacity = int(get_upgrade_value(fishing_upg, "FishingRod", levels["FishingRod"], "Effect_Value_Int"))
    ship_capacity = int(get_upgrade_value(fishing_upg, "Ship", levels["Ship"], "Effect_Value_Int"))
    seat_count = int(get_upgrade_value(restaurant_upg, "Max_Customer_Limit", levels["Max_Customer_Limit"], "Effect_Value_Int"))
    seat_count = max(seat_count, 1)

    active_rate_id = get_rate_id_for_player_grade(fishing_upg, levels["PlayerGrade"])
    rate_map, rarity_to_fish, fish_rarity = build_fishing_maps(tables)
    if active_rate_id not in rate_map:
        raise KeyError(f"Fishing_Rate_Data에 {active_rate_id}가 없습니다.")
    fish_to_recipe, recipe_price, recipe_yield = build_recipe_maps(tables["recipes"], levels, restaurant_upg)
    customer_pool = build_customer_pool(tables, levels)

    tickets = rod_capacity
    fish_inventory: Dict[str, int] = {}
    dish_inventory: Dict[str, int] = {}
    rarity_counter: Dict[str, int] = {r: 0 for r in RARITY_COLS}
    fish_counter: Dict[str, int] = {}
    recipe_counter: Dict[str, int] = {}

    fish_caught = 0
    fishing_sessions = 0
    tickets_spent = 0
    dishes_cooked = 0
    dishes_sold = 0
    gross_sales = 0.0
    tips_sales = 0.0
    customers_spawned = 0
    customers_completed = 0
    special_spawned = 0
    vip_spawned = 0
    restaurant_idle_seconds = 0
    peak_fish_inventory = 0
    peak_dish_stock = 0

    seats = [SeatState() for _ in range(seat_count)]

    next_charge_at = bait_seconds
    next_visit_at = None
    if tickets >= rod_capacity:
        next_visit_at = wait_after_full_seconds

    def total_fish_inventory() -> int:
        return sum(fish_inventory.values())

    def total_dish_inventory() -> int:
        return sum(dish_inventory.values())

    def can_spawn_customer() -> bool:
        return total_dish_inventory() > 0

    def convert_all_fish_to_dishes() -> None:
        nonlocal dishes_cooked, peak_dish_stock
        to_delete = []
        for fish_id, count in list(fish_inventory.items()):
            recipe_id = fish_to_recipe.get(fish_id)
            if not recipe_id:
                continue
            produced = count * recipe_yield[recipe_id]
            if produced > 0:
                dish_inventory[recipe_id] = dish_inventory.get(recipe_id, 0) + produced
                dishes_cooked += produced
            to_delete.append(fish_id)
        for fish_id in to_delete:
            fish_inventory.pop(fish_id, None)
        peak_dish_stock = max(peak_dish_stock, total_dish_inventory())

    def consume_random_dish() -> Optional[str]:
        available = [rid for rid, cnt in dish_inventory.items() if cnt > 0]
        if not available:
            return None
        rid = rng.choice(available)
        dish_inventory[rid] -= 1
        if dish_inventory[rid] <= 0:
            del dish_inventory[rid]
        return rid

    def start_customer_for_seat(seat: SeatState) -> None:
        nonlocal customers_spawned, special_spawned, vip_spawned
        customer = pick_customer(customer_pool, rng)
        seat.customer = customer
        seat.orders_completed = 0
        seat.phase = "walking_to_seat"
        seat.timer = HALF_TRIP_AT_V1 / max(customer.flow_velocity, 0.0001)
        customers_spawned += 1
        if customer.grade == "Special":
            special_spawned += 1
        elif customer.grade == "VIP":
            vip_spawned += 1

    def begin_order(seat: SeatState, order_num: int) -> None:
        recipe_id = consume_random_dish()
        if recipe_id is None:
            seat.phase = "walking_to_despawn"
            seat.timer = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
            return
        seat.orders_completed = order_num
        seat.phase = f"waiting_order_{order_num}"
        if order_num == 1:
            seat.timer = seat.customer.first_order_time
        elif order_num == 2:
            seat.timer = seat.customer.second_order_time
        else:
            seat.timer = seat.customer.third_order_time
        seat.current_recipe_id = recipe_id

    for t in range(1, total_seconds + 1):
        # fishing ticket charge
        if bait_seconds > 0 and t >= next_charge_at:
            while t >= next_charge_at:
                if tickets < rod_capacity:
                    tickets += 1
                    if tickets >= rod_capacity and next_visit_at is None:
                        next_visit_at = t + wait_after_full_seconds
                next_charge_at += bait_seconds

        # visit to fishing/restaurant scene after full charge wait
        if next_visit_at is not None and t >= next_visit_at:
            fishing_sessions += 1
            available_space = max(ship_capacity - total_fish_inventory(), 0)
            casts = min(tickets, available_space)
            rate_probs = rate_map[active_rate_id]
            for _ in range(casts):
                rarity, fish_id = pick_rarity_and_fish(rate_probs, rarity_to_fish, rng)
                fish_inventory[fish_id] = fish_inventory.get(fish_id, 0) + 1
                fish_counter[fish_id] = fish_counter.get(fish_id, 0) + 1
                rarity_counter[rarity] = rarity_counter.get(rarity, 0) + 1
                fish_caught += 1
            tickets -= casts
            tickets_spent += casts
            peak_fish_inventory = max(peak_fish_inventory, total_fish_inventory())
            convert_all_fish_to_dishes()
            next_visit_at = None
            if tickets >= rod_capacity:
                next_visit_at = t + wait_after_full_seconds

        # restaurant seat simulation
        if total_dish_inventory() <= 0:
            restaurant_idle_seconds += 1

        for seat in seats:
            if seat.timer > 0:
                seat.timer = max(0.0, seat.timer - 1.0)
                if seat.timer > 0:
                    continue

            if seat.phase == "ready_for_spawn":
                if can_spawn_customer():
                    start_customer_for_seat(seat)
            elif seat.phase == "walking_to_seat":
                begin_order(seat, 1)
            elif seat.phase == "waiting_order_1":
                seat.phase = "eating_1"
                seat.timer = seat.customer.first_eat_time
            elif seat.phase == "eating_1":
                price = recipe_price.get(seat.current_recipe_id, 0.0)
                gross_sales += price
                if rng.random() < seat.customer.tips_rate:
                    tips_sales += price * seat.customer.tips_multi
                dishes_sold += 1
                if total_dish_inventory() > 0 and rng.random() < seat.customer.second_order_rate:
                    begin_order(seat, 2)
                else:
                    seat.phase = "walking_to_despawn"
                    seat.timer = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
            elif seat.phase == "waiting_order_2":
                seat.phase = "eating_2"
                seat.timer = seat.customer.second_eat_time
            elif seat.phase == "eating_2":
                price = recipe_price.get(seat.current_recipe_id, 0.0)
                gross_sales += price
                if rng.random() < seat.customer.tips_rate:
                    tips_sales += price * seat.customer.tips_multi
                dishes_sold += 1
                if total_dish_inventory() > 0 and rng.random() < seat.customer.third_order_rate:
                    begin_order(seat, 3)
                else:
                    seat.phase = "walking_to_despawn"
                    seat.timer = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
            elif seat.phase == "waiting_order_3":
                seat.phase = "eating_3"
                seat.timer = seat.customer.third_eat_time
            elif seat.phase == "eating_3":
                price = recipe_price.get(seat.current_recipe_id, 0.0)
                gross_sales += price
                if rng.random() < seat.customer.tips_rate:
                    tips_sales += price * seat.customer.tips_multi
                dishes_sold += 1
                seat.phase = "walking_to_despawn"
                seat.timer = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
            elif seat.phase == "walking_to_despawn":
                customers_completed += 1
                seat.phase = "dishwashing"
                seat.timer = float(fixed_row["DishWashTime"])
                seat.customer = None
                seat.current_recipe_id = None
            elif seat.phase == "dishwashing":
                seat.phase = "spawn_delay"
                seat.timer = random_delay(tables["fixed"], rng)
            elif seat.phase == "spawn_delay":
                seat.phase = "ready_for_spawn"

        peak_dish_stock = max(peak_dish_stock, total_dish_inventory())

    remaining_tickets = tickets
    remaining_dishes = total_dish_inventory()
    fish_left_in_inventory = total_fish_inventory()
    total_sales = gross_sales + tips_sales

    if dishes_sold == 0 and fish_caught == 0:
        bottleneck = "낚시 방문 주기/인벤토리 병목"
    elif remaining_dishes == 0 and fish_caught > 0:
        bottleneck = "낚시 공급 부족"
    elif restaurant_idle_seconds < total_seconds * 0.1:
        bottleneck = "좌석/체류시간 병목"
    else:
        bottleneck = "손님 유입 병목"

    rarity_df = pd.DataFrame({"Rarity": list(rarity_counter.keys()), "Count": list(rarity_counter.values())})
    fish_df = pd.DataFrame(sorted(fish_counter.items(), key=lambda x: (-x[1], x[0])), columns=["Fish_ID", "Count"])

    recipe_counter = {}
    for fish_id, count in fish_counter.items():
        recipe_id = fish_to_recipe.get(fish_id)
        if recipe_id:
            recipe_counter[recipe_id] = recipe_counter.get(recipe_id, 0) + count * recipe_yield[recipe_id]
    recipe_df = pd.DataFrame(sorted(recipe_counter.items(), key=lambda x: (-x[1], x[0])), columns=["Recipe_ID", "Cooked_Count"])

    return SimSummary(
        fish_caught=fish_caught,
        fishing_sessions=fishing_sessions,
        tickets_spent=tickets_spent,
        fish_left_in_inventory=fish_left_in_inventory,
        dishes_cooked=dishes_cooked,
        dishes_sold=dishes_sold,
        gross_sales=gross_sales,
        tips_sales=tips_sales,
        total_sales=total_sales,
        customers_spawned=customers_spawned,
        customers_completed=customers_completed,
        special_spawned=special_spawned,
        vip_spawned=vip_spawned,
        peak_fish_inventory=peak_fish_inventory,
        peak_dish_stock=peak_dish_stock,
        remaining_tickets=remaining_tickets,
        remaining_dishes=remaining_dishes,
        restaurant_idle_seconds=restaurant_idle_seconds,
        bottleneck=bottleneck,
        active_rate_id=active_rate_id,
        rarity_breakdown=rarity_df,
        fish_breakdown=fish_df,
        recipe_breakdown=recipe_df,
    )


# ---------------- UI ----------------

def main() -> None:
    st.set_page_config(page_title="Balance Simulator", layout="wide")
    st.title("밸런스 시뮬레이터")
    st.caption("낚시 풀충전 대기 → 방문 시 전부 낚기 → 전부 요리 → 좌석별 손님 소비를 반영한 프로토타입")

    with st.sidebar:
        base_dir = st.text_input("데이터 파일 폴더", value=str(Path(__file__).resolve().parent))
        total_seconds = st.slider("시뮬레이션 시간(초)", 600, SECONDS_PER_DAY * 7, SECONDS_PER_DAY, 600)
        wait_after_full_seconds = st.slider("풀충전 후 대기시간(초)", 0, SECONDS_PER_DAY, 0, 60)
        seed = st.number_input("랜덤 시드", min_value=0, value=42, step=1)

    try:
        tables = load_all_data(base_dir)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

    fish_ranges = get_upgrade_ranges(tables["fishing_upg"], tables["fishing_settings"])
    rest_ranges = get_upgrade_ranges(tables["restaurant_upg"], tables["restaurant_settings"])

    col1, col2 = st.columns(2)
    levels: Dict[str, int] = {}
    with col1:
        st.subheader("Fishing")
        for key in ["PlayerGrade", "BaitMaking", "FishingRod", "Ship"]:
            if key in fish_ranges:
                lo, hi = fish_ranges[key]
                levels[key] = st.slider(key, lo, hi, lo)
    with col2:
        st.subheader("Restaurant")
        for key in [
            "Master_Lv", "Max_Customer_Limit", "Max_Spawn_Limit_1", "Max_Spawn_Limit_2", "Weight",
            "Bonus_Tips_Multi", "Bonus_Dish_Price_1", "Bonus_Dish_Price_2", "Bonus_Food_1", "Bonus_Food_2",
        ]:
            if key in rest_ranges:
                lo, hi = rest_ranges[key]
                levels[key] = st.slider(key, lo, hi, lo)

    levels = clamp_levels(levels)
    result = run_simulation(tables, levels, total_seconds, wait_after_full_seconds, int(seed))

    st.info(f"현재 PlayerGrade {levels['PlayerGrade']} → 참조 Rate_ID: **{result.active_rate_id}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총수익", f"{result.total_sales:,.0f}")
    m2.metric("총 판매량", f"{result.dishes_sold:,}")
    m3.metric("낚은 물고기", f"{result.fish_caught:,}")
    m4.metric("병목", result.bottleneck)

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Active Rate_ID", "Fishing Sessions", "Tickets Spent", "Remaining Tickets", "Fish Left Inventory",
                "Dishes Cooked", "Dishes Sold", "Remaining Dishes", "Gross Sales", "Tips Sales", "Total Sales",
                "Customers Spawned", "Customers Completed", "Special Spawned", "VIP Spawned",
                "Peak Fish Inventory", "Peak Dish Stock", "Restaurant Idle Seconds",
            ],
            "Value": [
                result.active_rate_id, result.fishing_sessions, result.tickets_spent, result.remaining_tickets,
                result.fish_left_in_inventory, result.dishes_cooked, result.dishes_sold, result.remaining_dishes,
                round(result.gross_sales, 2), round(result.tips_sales, 2), round(result.total_sales, 2),
                result.customers_spawned, result.customers_completed, result.special_spawned, result.vip_spawned,
                result.peak_fish_inventory, result.peak_dish_stock, result.restaurant_idle_seconds,
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("등급별 어획")
        st.dataframe(result.rarity_breakdown, use_container_width=True)
    with c2:
        st.subheader("물고기별 어획")
        st.dataframe(result.fish_breakdown, use_container_width=True)
    with c3:
        st.subheader("레시피 생산량")
        st.dataframe(result.recipe_breakdown, use_container_width=True)

    with st.expander("원본 데이터 미리보기"):
        tabs = st.tabs(["Fishing_UPG", "Rates", "Fish", "Restaurant_UPG", "Recipes", "Guest_Actions", "Guest_Tips"])
        with tabs[0]:
            st.dataframe(tables["fishing_upg"], use_container_width=True)
        with tabs[1]:
            st.dataframe(tables["rates"], use_container_width=True)
        with tabs[2]:
            st.dataframe(tables["fish"], use_container_width=True)
        with tabs[3]:
            st.dataframe(tables["restaurant_upg"], use_container_width=True)
        with tabs[4]:
            st.dataframe(tables["recipes"], use_container_width=True)
        with tabs[5]:
            st.dataframe(tables["guest_actions"], use_container_width=True)
        with tabs[6]:
            st.dataframe(tables["guest_tips"], use_container_width=True)


if __name__ == "__main__":
    main()
