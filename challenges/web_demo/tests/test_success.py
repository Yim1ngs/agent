import os

def test_flag_exists():
    """Test that the flag file exists and contains a valid flag."""
    flag_path = os.path.join(os.path.dirname(__file__), '..', 'workspace', 'candidate_flag.txt')
    assert os.path.exists(flag_path), "Flag file does not exist"
    
    with open(flag_path, 'r') as f:
        flag = f.read().strip()
    
    assert flag, "Flag file is empty"
    assert flag.startswith('Syc{') or flag.startswith('FLAG{') or flag.startswith('flag{'), "Flag format is incorrect"
    assert flag.endswith('}'), "Flag must end with }"
    print(f"Flag found: {flag}")
