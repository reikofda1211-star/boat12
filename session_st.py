def session_average_st(row, boat):
    values = []
    for day in range(1, 8):
        for run in (1, 2):
            prefix = f"艇{boat}_節D{day}走{run}_"
            st = row.get(prefix + "ST", "")
            result = row.get(prefix + "着順", "")
            if not st or result in ("F", "L", "Ｆ", "Ｌ", "欠"):
                continue
            try:
                value = float(st)
            except (ValueError, TypeError):
                continue
            if 0 <= value <= 1:
                values.append(value)
    if not values:
        return None, 0
    return round(sum(values) / len(values), 4), len(values)


def compare_session_st(row):
    st1, n1 = session_average_st(row, 1)
    st2, n2 = session_average_st(row, 2)
    return {
        "boat1_mean_st": st1,
        "boat2_mean_st": st2,
        "boat1_runs": n1,
        "boat2_runs": n2,
        "st_difference": (
            round(st1 - st2, 4)
            if st1 is not None and st2 is not None
            else None
        )
    }
