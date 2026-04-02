from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

SECONDS_PER_DAY = 24 * 60 * 60
HALF_TRIP_AT_V1 = 17.61 / 2.0
RARITY_COLS = ["Trash", "Normal", "Fine", "Superior", "Rare", "Elite", "Fantastic", "Legendary"]


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
        self.has_eaten = False 
        self.current_recipe_id: Optional[str] = None


@dataclass
class SimSummary:
    fish_caught: float
    fishing_sessions: float
    tickets_spent: float
    fish_left_in_inventory: float
    dishes_cooked: float
    dishes_discarded: float
    dishes_sold: float
    gross_sales: float
    tips_sales: float
    total_sales: float
    customers_spawned: float
    customers_completed: float
    special_spawned: float
    vip_spawned: float
    peak_fish_inventory: float
    peak_dish_stock: float
    remaining_tickets: float
    remaining_dishes: float
    restaurant_idle_seconds: float
    out_of_stock_seconds: float
    bottleneck: str
    active_rate_id: str
    rarity_breakdown: pd.DataFrame
    fish_breakdown: pd.DataFrame
    recipe_breakdown: pd.DataFrame


# ---------------- data loading ----------------

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(how="all").reset_index(drop=True)

    header_idx = 0
    keywords = ["Upgrade_Type", "Fish_ID", "Recipe_ID", "Customer_ID", "Gacha_Group_ID", "DishWashTime", "Upgrade", "Type", "Effect_Value_Int", "Level", "Cost"]
    for i in range(min(10, len(out))):
        row_values = [str(x).strip() for x in out.iloc[i].tolist()]
        if any(k in v for k in keywords for v in row_values):
            header_idx = i
            break

    raw_columns = [str(c).replace('\n', '').replace('\r', '').strip() for c in out.iloc[header_idx].tolist()]
    out.columns = raw_columns
    out = out.iloc[header_idx + 1:].reset_index(drop=True)
    out = out.dropna(how="all")

    for col in out.columns:
        clean_col = col.replace(" ", "").replace("_", "").lower()
        if clean_col == "upgradetype":
            out.rename(columns={col: "Upgrade_Type"}, inplace=True)
        elif clean_col == "level":
            out.rename(columns={col: "Level"}, inplace=True)

    first_col_values_to_drop = {"분류", "아이디", "손님 ID", "등급", "이름", "설거지 소요 시간", "Upgrade_Type", "Recipe_ID", "Fish_ID"}
    row_tokens_to_drop = {"분류", "레벨", "비용", "등급", "이름", "Int", "Float", "Enum", "String", "string", "int", "float", "enum", "자료형", "계산 방식", "아이디", "텍스트 ID", "다음단계필요재화", "값", "참조아이디", "효과 값 1", "효과 값 2", "설거지 소요 시간", "최소 입장 딜레이", "최대 입장 딜레이", "한글 이름", "영어 이름", "가중치", "출입 속도", "첫 번째 주문 시간", "첫 번째 먹는 시간", "두 번째 주문 확률", "두 번째 주문 시간", "두 번째 먹는 시간", "세 번째 주문 확률", "세 번째 주문 시간", "세 번째 먹는 시간", "팁 확률", "팁 배수", "정가", "생산량", "재료"}

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
def load_all_data(fishing_id: str, restaurant_id: str, guest_id: str) -> Dict[str, pd.DataFrame]:
    fishing_url = f"https://docs.google.com/spreadsheets/d/{fishing_id}/export?format=xlsx"
    restaurant_url = f"https://docs.google.com/spreadsheets/d/{restaurant_id}/export?format=xlsx"
    guest_url = f"https://docs.google.com/spreadsheets/d/{guest_id}/export?format=xlsx"

    fishing_sheets = pd.read_excel(fishing_url, sheet_name=None, header=None)
    restaurant_sheets = pd.read_excel(restaurant_url, sheet_name=None, header=None)
    guest_sheets = pd.read_excel(guest_url, sheet_name=None, header=None)

    tables = {
        "restaurant_upg": clean_table(restaurant_sheets["Restaurant_UPG_Data"]),
        "restaurant_settings": clean_table(restaurant_sheets["Restaurant_UPG_Setting"]),
        "recipes": clean_table(restaurant_sheets["Recipe_Data"]),
        "fishing_upg": clean_table(fishing_sheets["Fishing_UPG_Data"]),
        "fishing_settings": clean_table(fishing_sheets["Fishing_UPG_Setting"]),
        "rates": clean_table(fishing_sheets["Fishing_Rate_Data"]),
        "fish": clean_table(fishing_sheets["Fish_Data"]),
        "fixed": clean_table(guest_sheets["Fixed_Value"]),
        "guest_actions": clean_table(guest_sheets["Customer_Action"]),
        "guest_tips": clean_table(guest_sheets["Customer_Tips"]),
    }
    
    if "Menu_UPG_Data" in restaurant_sheets:
        tables["menu_upg"] = clean_table(restaurant_sheets["Menu_UPG_Data"])
    else:
        tables["menu_upg"] = pd.DataFrame(columns=["Level", "Effect_Value_Int"])

    return tables


# ---------------- helpers ----------------

def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def get_upgrade_ranges(upg_df: pd.DataFrame, settings_df: pd.DataFrame) -> Dict[str, tuple[int, int]]:
    if "Upgrade_Type" not in upg_df.columns or "Level" not in upg_df.columns:
        st.error(f"🚨 **[데이터 로드 오류]** 시트에서 'Upgrade_Type' 또는 'Level' 컬럼을 찾지 못했습니다.")
        st.info(f"👉 **파이썬이 읽어들인 실제 컬럼명:** {upg_df.columns.tolist()}")
        st.warning("구글 시트의 첫 번째 줄(헤더)에 오타가 있거나, '분류' 같은 한글로 되어있는지 확인해 주세요!")
        st.stop() 

    settings_df = to_numeric(settings_df, ["Max_Level"])
    if "Upgrade_Type" not in settings_df.columns:
        max_from_settings = {}
    else:
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
        get_upgrade_value(restaurant_upg_df, "Bonus_Food_1", levels.get("Bonus_Food_1", 1), "Effect_Value_Int")
        + get_upgrade_value(restaurant_upg_df, "Bonus_Food_2", levels.get("Bonus_Food_2", 1), "Effect_Value_Int")
    )
    price_bonus = (
        get_upgrade_value(restaurant_upg_df, "Bonus_Dish_Price_1", levels.get("Bonus_Dish_Price_1", 1), "Effect_Value_Float")
        + get_upgrade_value(restaurant_upg_df, "Bonus_Dish_Price_2", levels.get("Bonus_Dish_Price_2", 1), "Effect_Value_Float")
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
        tips_by_grade[str(row["Grade"]).strip()] = {
            "rate": normalize_rate(row.get("Tips_Rate") or 0.0),
            "multi": float(row.get("Tips_Multi") or 0.0),
        }

    special_weight_bonus = float(get_upgrade_value(tables["restaurant_upg"], "Weight", levels.get("Weight", 1), "Effect_Value_Int"))
    tip_bonus_multi = float(get_upgrade_value(tables["restaurant_upg"], "Bonus_Tips_Multi", levels.get("Bonus_Tips_Multi", 1), "Effect_Value_Float"))

    pool: List[CustomerProfile] = []
    for _, row in actions.iterrows():
        try:
            cid = int(float(row["Customer_ID"]))
        except ValueError:
            continue
            
        grade = str(row["Grade"]).strip()
        weight = float(row.get("Weight") or 0.0)
        
        if grade in ["Special", "VIP"]:
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


def pick_customer(pool: List[CustomerProfile], rng: random.Random, 
                  current_special: int, max_special: int, 
                  current_vip: int, max_vip: int) -> CustomerProfile:
    available_pool = []
    
    for c in pool:
        if c.grade == "Special" and current_special >= max_special:
            continue
        if c.grade == "VIP" and current_vip >= max_vip:
            continue
        available_pool.append(c)

    if not available_pool:
        available_pool = pool
        
    weights = [c.weight for c in available_pool]
    return rng.choices(available_pool, weights=weights, k=1)[0]


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

    bait_seconds = int(get_upgrade_value(fishing_upg, "BaitMaking", levels.get("BaitMaking", 1), "Effect_Value_Int"))
    rod_capacity = int(get_upgrade_value(fishing_upg, "FishingRod", levels.get("FishingRod", 1), "Effect_Value_Int"))
    ship_capacity = int(get_upgrade_value(fishing_upg, "Ship", levels.get("Ship", 1), "Effect_Value_Int"))
    seat_count = int(get_upgrade_value(restaurant_upg, "Max_Customer_Limit", levels.get("Max_Customer_Limit", 1), "Effect_Value_Int"))
    seat_count = max(seat_count, 1)

    max_special_limit = int(get_upgrade_value(restaurant_upg, "Max_Spawn_Limit_1", levels.get("Max_Spawn_Limit_1", 1), "Effect_Value_Int"))
    max_vip_limit = int(get_upgrade_value(restaurant_upg, "Max_Spawn_Limit_2", levels.get("Max_Spawn_Limit_2", 1), "Effect_Value_Int"))

    menu_upg = tables.get("menu_upg", pd.DataFrame())
    if not menu_upg.empty and "Level" in menu_upg.columns and "Effect_Value_Int" in menu_upg.columns:
        menu_df = to_numeric(menu_upg, ["Level", "Effect_Value_Int"])
        menu_row = menu_df[menu_df["Level"] == levels.get("Menu_Level", 1)]
        max_menu_slots = int(menu_row.iloc[0]["Effect_Value_Int"]) if not menu_row.empty else 2
    else:
        max_menu_slots = 2

    active_rate_id = get_rate_id_for_player_grade(fishing_upg, levels.get("PlayerGrade", 1))
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
    dishes_discarded = 0
    dishes_sold = 0
    gross_sales = 0.0
    tips_sales = 0.0
    customers_spawned = 0
    customers_completed = 0
    special_spawned = 0
    vip_spawned = 0
    restaurant_idle_seconds = 0
    out_of_stock_seconds = 0
    peak_fish_inventory = 0
    peak_dish_stock = 0
    reserved_dishes = 0

    seats = [SeatState() for _ in range(seat_count)]
    exiting_timers: List[Tuple[float, bool]] = []

    next_charge_at = bait_seconds
    next_visit_at = None
    if tickets >= rod_capacity:
        next_visit_at = wait_after_full_seconds

    def total_fish_inventory() -> int:
        return sum(fish_inventory.values())

    def total_dish_inventory() -> int:
        return sum(dish_inventory.values())

    def can_spawn_customer() -> bool:
        return (total_dish_inventory() - reserved_dishes) > 0

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

    def manage_menu_and_discard() -> None:
        nonlocal dishes_discarded
        available_rids = [rid for rid, cnt in dish_inventory.items() if cnt > 0]
        if len(available_rids) <= max_menu_slots:
            return
            
        available_rids.sort(key=lambda x: recipe_price.get(x, 0.0) * dish_inventory[x], reverse=True)
        
        discard_rids = available_rids[max_menu_slots:]
        for rid in discard_rids:
            dishes_discarded += dish_inventory[rid]
            del dish_inventory[rid]

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
        nonlocal customers_spawned, special_spawned, vip_spawned, reserved_dishes
        reserved_dishes += 1  
        
        current_special = sum(1 for s in seats if s.customer and s.customer.grade == "Special")
        current_vip = sum(1 for s in seats if s.customer and s.customer.grade == "VIP")
        
        customer = pick_customer(customer_pool, rng, current_special, max_special_limit, current_vip, max_vip_limit)
        
        seat.customer = customer
        seat.orders_completed = 0
        seat.has_eaten = False 
        seat.phase = "walking_to_seat"
        seat.timer = HALF_TRIP_AT_V1 / max(customer.flow_velocity, 0.0001)
        
        customers_spawned += 1
        if customer.grade == "Special":
            special_spawned += 1
        elif customer.grade == "VIP":
            vip_spawned += 1

    def begin_order(seat: SeatState, order_num: int) -> None:
        nonlocal reserved_dishes
        
        if order_num == 1:
            reserved_dishes -= 1
            
        recipe_id = consume_random_dish()
        
        if recipe_id is None:
            exit_time = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
            exiting_timers.append((exit_time, False))
            
            seat.phase = "spawn_delay"
            seat.timer = random_delay(tables["fixed"], rng)
            seat.customer = None
            return
            
        seat.has_eaten = True
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
        if bait_seconds > 0 and t >= next_charge_at:
            while t >= next_charge_at:
                if tickets < rod_capacity:
                    tickets += 1
                    if tickets >= rod_capacity and next_visit_at is None:
                        next_visit_at = t + wait_after_full_seconds
                next_charge_at += bait_seconds

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
            manage_menu_and_discard() 
            
            next_visit_at = None
            if tickets >= rod_capacity:
                next_visit_at = t + wait_after_full_seconds

        new_exiting = []
        for t_exit, has_eaten in exiting_timers:
            t_exit -= 1.0
            if t_exit <= 0:
                if has_eaten:
                    customers_completed += 1
            else:
                new_exiting.append((t_exit, has_eaten))
        exiting_timers = new_exiting

        if total_dish_inventory() <= 0:
            out_of_stock_seconds += 1
            
        if all(s.customer is None for s in seats):
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
                    exit_time = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
                    exiting_timers.append((exit_time, True))
                    seat.phase = "dishwashing"
                    seat.timer = float(fixed_row["DishWashTime"])
                    seat.customer = None
                    seat.current_recipe_id = None
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
                    exit_time = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
                    exiting_timers.append((exit_time, True))
                    seat.phase = "dishwashing"
                    seat.timer = float(fixed_row["DishWashTime"])
                    seat.customer = None
                    seat.current_recipe_id = None
            elif seat.phase == "waiting_order_3":
                seat.phase = "eating_3"
                seat.timer = seat.customer.third_eat_time
            elif seat.phase == "eating_3":
                price = recipe_price.get(seat.current_recipe_id, 0.0)
                gross_sales += price
                if rng.random() < seat.customer.tips_rate:
                    tips_sales += price * seat.customer.tips_multi
                dishes_sold += 1
                
                exit_time = HALF_TRIP_AT_V1 / max(seat.customer.flow_velocity, 0.0001)
                exiting_timers.append((exit_time, True))
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
    elif out_of_stock_seconds > total_seconds * 0.1 and remaining_dishes == 0:
        bottleneck = "낚시 공급 부족 (재고 고갈)"
    elif restaurant_idle_seconds < total_seconds * 0.1:
        bottleneck = "좌석/체류시간 병목 (회전율 한계)"
    else:
        bottleneck = "손님 유입 병목 (마케팅/스폰 부족)"

    rarity_df = pd.DataFrame({"Rarity": list(rarity_counter.keys()), "Count": list(rarity_counter.values())})
    fish_df = pd.DataFrame(list(fish_counter.items()), columns=["Fish_ID", "Count"])

    recipe_counter = {}
    for fish_id, count in fish_counter.items():
        recipe_id = fish_to_recipe.get(fish_id)
        if recipe_id:
            recipe_counter[recipe_id] = recipe_counter.get(recipe_id, 0) + count * recipe_yield[recipe_id]
            
    recipe_df = pd.DataFrame(list(recipe_counter.items()), columns=["Recipe_ID", "Cooked_Count"])

    return SimSummary(
        fish_caught=float(fish_caught),
        fishing_sessions=float(fishing_sessions),
        tickets_spent=float(tickets_spent),
        fish_left_in_inventory=float(fish_left_in_inventory),
        dishes_cooked=float(dishes_cooked),
        dishes_discarded=float(dishes_discarded),
        dishes_sold=float(dishes_sold),
        gross_sales=float(gross_sales),
        tips_sales=float(tips_sales),
        total_sales=float(total_sales),
        customers_spawned=float(customers_spawned),
        customers_completed=float(customers_completed),
        special_spawned=float(special_spawned),
        vip_spawned=float(vip_spawned),
        peak_fish_inventory=float(peak_fish_inventory),
        peak_dish_stock=float(peak_dish_stock),
        remaining_tickets=float(remaining_tickets),
        remaining_dishes=float(remaining_dishes),
        restaurant_idle_seconds=float(restaurant_idle_seconds),
        out_of_stock_seconds=float(out_of_stock_seconds),
        bottleneck=bottleneck,
        active_rate_id=active_rate_id,
        rarity_breakdown=rarity_df,
        fish_breakdown=fish_df,
        recipe_breakdown=recipe_df,
    )


def sort_by_numeric_id(item):
    try:
        return (0, int(item))
    except ValueError:
        return (1, str(item))


def run_multiple_simulations(tables: Dict[str, pd.DataFrame], levels: Dict[str, int], total_seconds: int, wait_after_full_seconds: int, base_seed: int, num_runs: int = 10) -> SimSummary:
    summaries = []
    for i in range(num_runs):
        summaries.append(run_simulation(tables, levels, total_seconds, wait_after_full_seconds, base_seed + i))

    def get_avg(field: str) -> float:
        return sum(getattr(s, field) for s in summaries) / num_runs

    bottlenecks = [s.bottleneck for s in summaries]
    common_bottleneck = Counter(bottlenecks).most_common(1)[0][0]

    rarity_df = pd.concat([s.rarity_breakdown for s in summaries]).groupby("Rarity", as_index=False)["Count"].mean()
    
    fish_concat = pd.concat([s.fish_breakdown for s in summaries])
    if not fish_concat.empty:
        fish_df = fish_concat.groupby("Fish_ID", as_index=False)["Count"].sum()
        fish_df["Count"] = fish_df["Count"] / num_runs
        fish_df["_sort_key"] = fish_df["Fish_ID"].apply(sort_by_numeric_id)
        fish_df = fish_df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    else:
        fish_df = fish_concat

    recipe_concat = pd.concat([s.recipe_breakdown for s in summaries])
    if not recipe_concat.empty:
        recipe_df = recipe_concat.groupby("Recipe_ID", as_index=False)["Cooked_Count"].sum()
        recipe_df["Cooked_Count"] = recipe_df["Cooked_Count"] / num_runs 
        recipe_df["_sort_key"] = recipe_df["Recipe_ID"].apply(sort_by_numeric_id)
        recipe_df = recipe_df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    else:
        recipe_df = recipe_concat

    return SimSummary(
        fish_caught=get_avg("fish_caught"),
        fishing_sessions=get_avg("fishing_sessions"),
        tickets_spent=get_avg("tickets_spent"),
        fish_left_in_inventory=get_avg("fish_left_in_inventory"),
        dishes_cooked=get_avg("dishes_cooked"),
        dishes_discarded=get_avg("dishes_discarded"),
        dishes_sold=get_avg("dishes_sold"),
        gross_sales=get_avg("gross_sales"),
        tips_sales=get_avg("tips_sales"),
        total_sales=get_avg("total_sales"),
        customers_spawned=get_avg("customers_spawned"),
        customers_completed=get_avg("customers_completed"),
        special_spawned=get_avg("special_spawned"),
        vip_spawned=get_avg("vip_spawned"),
        peak_fish_inventory=get_avg("peak_fish_inventory"),
        peak_dish_stock=get_avg("peak_dish_stock"),
        remaining_tickets=get_avg("remaining_tickets"),
        remaining_dishes=get_avg("remaining_dishes"),
        restaurant_idle_seconds=get_avg("restaurant_idle_seconds"),
        out_of_stock_seconds=get_avg("out_of_stock_seconds"),
        bottleneck=common_bottleneck,
        active_rate_id=summaries[0].active_rate_id,
        rarity_breakdown=rarity_df,
        fish_breakdown=fish_df,
        recipe_breakdown=recipe_df,
    )


# ---------------- UI ----------------

def extract_id(url: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else ""


def main() -> None:
    st.set_page_config(page_title="Balance Simulator", layout="wide")
    st.title("밸런스 시뮬레이터")
    st.caption("낚시 풀충전 대기 → 방문 시 전부 낚기 → 전부 요리 → 좌석별 손님 소비를 반영한 프로토타입")

    with st.sidebar:
        st.subheader("Google Sheets Links")
        fishing_link = st.text_input("Fishing Data URL", value="https://docs.google.com/spreadsheets/d/1hRIRi-KPGhrhNpjivmk3fK_2HeK9YeP1eK5ygCh837w/edit?usp=drive_link")
        restaurant_link = st.text_input("Restaurant Data URL", value="https://docs.google.com/spreadsheets/d/1iJgw7DdrnDBxqxrpiy8QH0hCfX3CR_MbIl6RgIpvS7s/edit?usp=drive_link")
        guest_link = st.text_input("Customer Data URL", value="https://docs.google.com/spreadsheets/d/1YO5eyJvc26dD0JXTYaJRWGz3gd9cQnqkOjO23i0SKVc/edit?usp=drive_link")

        st.divider()
        total_seconds = st.number_input("시뮬레이션 시간(초)", min_value=600, max_value=SECONDS_PER_DAY * 7, value=SECONDS_PER_DAY, step=600)
        wait_after_full_seconds = st.slider("풀충전 후 대기시간(초)", 0, SECONDS_PER_DAY, 0, 60)
        num_runs = st.number_input("시뮬레이션 반복 횟수 (평균)", min_value=1, max_value=100, value=10, step=1)
        base_seed = st.number_input("시작 랜덤 시드", min_value=0, value=42, step=1)

    fishing_id = extract_id(fishing_link)
    restaurant_id = extract_id(restaurant_link)
    guest_id = extract_id(guest_link)

    try:
        with st.spinner("구글 시트에서 데이터를 불러오는 중..."):
            tables = load_all_data(fishing_id, restaurant_id, guest_id)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.warning("팁: 구글 시트의 공유 권한이 '링크가 있는 모든 사용자(뷰어)'로 설정되어 있는지 확인해주세요.")
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
        
        if "menu_upg" in tables and not tables["menu_upg"].empty:
            menu_df = to_numeric(tables["menu_upg"], ["Level"])
            menu_max = menu_df["Level"].max() if not menu_df["Level"].dropna().empty else 9
            levels["Menu_Level"] = st.slider("Menu_Level (메뉴 슬롯 제한)", 1, int(menu_max), 1)
        else:
            levels["Menu_Level"] = 1

        for key in [
            "Master_Lv", "Max_Customer_Limit", "Max_Spawn_Limit_1", "Max_Spawn_Limit_2", "Weight",
            "Bonus_Tips_Multi", "Bonus_Dish_Price_1", "Bonus_Dish_Price_2", "Bonus_Food_1", "Bonus_Food_2",
        ]:
            if key in rest_ranges:
                lo, hi = rest_ranges[key]
                levels[key] = st.slider(key, lo, hi, lo)

    levels = clamp_levels(levels)
    
    with st.spinner(f"시뮬레이션을 {num_runs}회 반복 실행하며 평균을 계산 중입니다..."):
        result = run_multiple_simulations(tables, levels, total_seconds, wait_after_full_seconds, int(base_seed), int(num_runs))

    st.info(f"현재 PlayerGrade {levels.get('PlayerGrade', 1)} → 참조 Rate_ID: **{result.active_rate_id}** (총 {num_runs}회 평균 데이터)")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("총수익 (평균)", f"{result.total_sales:,.0f}")
    m2.metric("팁 수익 (평균)", f"{result.tips_sales:,.0f}")
    m3.metric("총 판매량 (평균)", f"{result.dishes_sold:,.1f}")
    m4.metric("폐기된 음식 (평균)", f"{result.dishes_discarded:,.1f}")  
    m5.metric("낚은 물고기 (평균)", f"{result.fish_caught:,.1f}")
    m6.metric("주요 병목 현상", result.bottleneck)

    st.markdown("---")
    st.subheader(f"손닙 방문 통계 (평균)")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    normal_spawned = result.customers_spawned - result.special_spawned - result.vip_spawned
    
    c_m1.metric("총 방문객", f"{result.customers_spawned:,.1f}")
    c_m2.metric("일반 손님 (Normal)", f"{normal_spawned:,.1f}")
    c_m3.metric("특수 손님 (Special)", f"{result.special_spawned:,.1f}")
    c_m4.metric("VIP 손님", f"{result.vip_spawned:,.2f}")
    st.markdown("---")

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Active Rate_ID", "Fishing Sessions", "Tickets Spent", "Remaining Tickets", "Fish Left Inventory",
                "Dishes Cooked", "Dishes Sold", "Dishes Discarded", "Remaining Dishes", "Gross Sales", "Tips Sales", "Total Sales",
                "Customers Spawned", "Customers Completed", "Special Spawned", "VIP Spawned",
                "Peak Fish Inventory", "Peak Dish Stock", "Out of Stock Seconds", "Restaurant Idle Seconds",
            ],
            "Value": [
                result.active_rate_id, 
                round(result.fishing_sessions, 2), 
                round(result.tickets_spent, 2), 
                round(result.remaining_tickets, 2),
                round(result.fish_left_in_inventory, 2), 
                round(result.dishes_cooked, 2), 
                round(result.dishes_sold, 2), 
                round(result.dishes_discarded, 2),
                round(result.remaining_dishes, 2),
                round(result.gross_sales, 2), 
                round(result.tips_sales, 2), 
                round(result.total_sales, 2),
                round(result.customers_spawned, 2), 
                round(result.customers_completed, 2), 
                round(result.special_spawned, 2), 
                round(result.vip_spawned, 2),
                round(result.peak_fish_inventory, 2), 
                round(result.peak_dish_stock, 2), 
                round(result.out_of_stock_seconds, 2), 
                round(result.restaurant_idle_seconds, 2),
            ],
        }
    )
    with st.expander("시뮬레이션 상세 요약 데이터 보기 (10회 평균값)"):
        st.dataframe(summary_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("등급별 어획 (평균)")
        st.dataframe(result.rarity_breakdown.round(1), use_container_width=True)
    with c2:
        st.subheader("물고기별 어획 (평균)")
        st.dataframe(result.fish_breakdown.round(1), use_container_width=True)
    with c3:
        st.subheader("레시피 생산량 (평균)")
        st.dataframe(result.recipe_breakdown.round(1), use_container_width=True)

    with st.expander("원본 데이터 미리보기"):
        tabs = st.tabs(["Fishing_UPG", "Rates", "Fish", "Restaurant_UPG", "Menu_UPG", "Recipes", "Guest_Actions", "Guest_Tips"])
        with tabs[0]:
            st.dataframe(tables["fishing_upg"], use_container_width=True)
        with tabs[1]:
            st.dataframe(tables["rates"], use_container_width=True)
        with tabs[2]:
            st.dataframe(tables["fish"], use_container_width=True)
        with tabs[3]:
            st.dataframe(tables["restaurant_upg"], use_container_width=True)
        with tabs[4]:
            st.dataframe(tables["menu_upg"], use_container_width=True)
        with tabs[5]:
            st.dataframe(tables["recipes"], use_container_width=True)
        with tabs[6]:
            st.dataframe(tables["guest_actions"], use_container_width=True)
        with tabs[7]:
            st.dataframe(tables["guest_tips"], use_container_width=True)


if __name__ == "__main__":
    main()