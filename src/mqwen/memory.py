"""Conservative Metal residency budget; this is a cap, not an allocation."""

def residency_budget(info):
    physical = int(info.get('memory_size',0))
    recommended = int(info.get('max_recommended_working_set_size',0))
    # Leave at least 40% of physical memory outside this process's wired budget.
    return max(0,min(recommended,physical*3//5))
