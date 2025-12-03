# For testing what would be uniqui set of pair of single qubit gate and 
# CNOT gate in either direction. Equivalent gate sets mean here that
# Gatesets have same output too al paulipairs in terms of do they produce I
# on either qubit or both qubits.
# 
# This means that agte sets would have same effect on distance of pauli gadget.
# 
# One unieuq set is I, V SV for control qubit and V, S, VS for target qubit.
#

def apply_v(p): # z,z xor x
    phase = 1
    if p == 0b10:
        phase = -1
    return (0b10 & p) | ((0b01 & p) ^ (p >> 1)), phase

def apply_s(p): # z xor x, x
    phase = 1
    if p == 0b11:
        phase = -1
    return 0b01 & p | ((0b10 & p) ^ ((p & 0b01) << 1)), phase

def apply_I(p):
    return p, 1

def apply_sv(p):
    pauli,phase1 = apply_s(p)
    pauli,phase2 = apply_v(pauli)
    return pauli, phase1*phase2

def apply_vs(p):
    pauli,phase1 = apply_v(p)
    pauli,phase2 = apply_s(pauli)
    return pauli, phase1*phase2

def apply_svs(p):
    pauli,phase1 = apply_s(p)
    pauli,phase2 = apply_v(pauli)
    pauli,phase3 = apply_s(pauli)
    return pauli, phase1*phase2*phase3

def apply_h(p):
    return apply_svs(p)

def apply_cnot(p1, p2):
    phase = 1
    if (p1 == 0b01 and p2 == 0b10) or (p1 == 0b11 and p2 == 0b11):
        phase = -1
    pauli1 = (p1 & 0b01) | ((p1 ^ p2) & 0b10)
    pauli2 = ((p1 ^ p2) & 0b01) | (p2 & 0b10)
    return pauli1, pauli2, phase

def check_equal_gates():
    print('Check all sets are included in reduced set')
    gate_set_control = [apply_I, apply_v, apply_sv]
    gate_set_target = [apply_v, apply_s, apply_vs]
    cnot_reversed_options = [False]
#    gate_set_control = [apply_I, apply_h, apply_s]
#    gate_set_target = [apply_I, apply_h, apply_s]
#    cnot_reversed_options = [True, False]
    missing = 0
    # for each possible gate set, check that there is one in reduced set having same effect
    for op11 in [apply_I, apply_v, apply_s, apply_h, apply_sv, apply_sv]:
        for op12 in [apply_I, apply_v, apply_s, apply_h, apply_sv, apply_sv]:
            for cnotr1 in [True, False]:
                # Try to match one with reduced gate set
                found_combination = False
                for op21 in gate_set_control:
                    for op22 in gate_set_target:
                        for cnotr2 in cnot_reversed_options:
                            # This gate set is in in the reduceed set directly
                            if op11 == op21 and op12 == op22 and cnotr1 == cnotr2:
                                found_combination = True
                            else:
                                this_is_ok = True
                                for p1 in [0b01,0b11,0b10,0b00]:
                                    for p2 in [0b01,0b11,0b10,0b00]:
                                        p11,p12,_ = apply_cnot(op11(p1)[0], op12(p2)[0])
                                        p21,p22,_ = apply_cnot(op21(p1)[0], op22(p2)[0])
                                        if cnotr1:
                                            p12,p11,_ = apply_cnot(op12(p2)[0], op11(p1)[0])
                                        if cnotr2:
                                            p22,p21,_ = apply_cnot(op22(p2)[0], op21(p1)[0])
                                        if ((p11==0b00) != (p21==0b00)) or ((p12==0b00) != (p22==0b00)):
                                            this_is_ok = False
                                if this_is_ok:
                                    found_combination = True
#                                    print('Found match for:', op11.__name__, op12.__name__, cnotr1, '==', op21.__name__, op22.__name__, cnotr2)
                            if found_combination:
                                break
                        if found_combination:
                            break
                    if found_combination:
                        break
                if not found_combination:
                    missing += 1
                    print('ERROR No match found for:', op11.__name__, op12.__name__, cnotr1)
    print(f'Missing sets found: {missing}')
    print('Now checking duplicates in reduced set')
    duplicates = 0
    # Check that proposed gate sets does not inlucde equal sets
    for op11 in gate_set_control:
        for op12 in gate_set_target:
            for cnotr1 in cnot_reversed_options:
                # Try to match one with reduced gate set
                found_combination = False
                for op21 in gate_set_control:
                    for op22 in gate_set_target:
                        for cnotr2 in cnot_reversed_options:
                            if op11 == op21 and op12 == op22 and cnotr1 == cnotr2:
                                continue
                            # This gate set is in in the reduceed set directly
                            similar = True
                            for p1 in [0b01,0b11,0b10,0b00]:
                                for p2 in [0b01,0b11,0b10,0b00]:
                                    p11,p12,_ = apply_cnot(op11(p1)[0], op12(p2)[0])
                                    p21,p22,_ = apply_cnot(op21(p1)[0], op22(p2)[0])
                                    if cnotr1:
                                        p12,p11,_ = apply_cnot(op12(p2)[0], op11(p1)[0])
                                    if cnotr2:
                                        p22,p21,_ = apply_cnot(op22(p2)[0], op21(p1)[0])
                                    if ((p11==0b00) != (p21==0b00)) or ((p12==0b00) != (p22==0b00)):
                                        similar = False
                            if similar:
                                duplicates += 1
                                print('ERROR Duplicate match found:', op11.__name__, op12.__name__, cnotr1,
                                      'and', op21.__name__, op22.__name__, cnotr2)
    print(f'Duplicate sets found: {int(duplicates/2)}')
    print('test complete')

check_equal_gates()