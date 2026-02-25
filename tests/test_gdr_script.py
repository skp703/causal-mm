
from causal_mm.fcm import FCMGraph, Concept, EdgeEstimate
from causal_mm.metrics import generalized_distance_ratio

def test_gdr_calculation():
    # Create dummy FCMs
    c1 = Concept(id="1")
    c2 = Concept(id="2")

    fcm_a = FCMGraph(concepts=[c1, c2], edges=[])
    fcm_b = FCMGraph(concepts=[c1, c2], edges=[])

    # A: 1->2 = 0.5
    fcm_a.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", scaled_weight=0.5)

    # B: 1->2 = 0.8 (Same sign)
    fcm_b.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", scaled_weight=0.8)

    # Parameters
    gamma = 2.0
    delta = 0.0
    epsilon = 2.0
    alpha = 1.0
    beta = 1.0
    gamma_prime = 1.0

    # Calculation
    # p_c = 2, p_uA = 0, p_uB = 0
    # Numerator:
    # 1->2: |0.5 - 0.8| * alpha = 0.3
    # Others: 0
    # Total Num = 0.3
    
    # Denominator:
    # Term1 = (epsilon*beta + delta) * p_c^2 = (2*1 + 0) * 4 = 8
    # Term2 = gamma_prime * (0) = 0
    # Term3 = alpha * ( (epsilon*beta + delta)*p_c + gamma_prime*(0) ) = 1 * (2*2) = 4
    # Denom = 8 + 0 - 4 = 4
    
    # Expected GDR = 0.3 / 4 = 0.075

    gdr = generalized_distance_ratio(fcm_a, fcm_b)
    print(f"Calculated GDR: {gdr}")
    
    assert abs(gdr - 0.075) < 1e-6, f"Expected 0.075, got {gdr}"

if __name__ == "__main__":
    test_gdr_calculation()
