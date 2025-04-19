import streamlit as st
import hashlib
import re
import os
import pandas as pd
import pickle
from collections import Counter

# Set page configuration
st.set_page_config(page_title="IoT Malware Detector", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    # In a real app, you would load a pre-trained model
    # This is a simplified model for demonstration
    st.session_state.model = pickle.dumps({
        "malware_strings": ["suspicious_function", "botnet", "ddos", 
                           "shell_exec", "system_call", "remote_shell",
                           "connect_back", "persistence", "evasion"],
        "malware_types": {
            "botnet": ["botnet", "cnc", "command_and_control", "irc_bot", "zombie_net"],
            "ransomware": ["encrypt", "ransom", "payment", "bitcoin", "decrypt_key"],
            "backdoor": ["backdoor", "remote_shell", "reverse_shell", "connect_back", "persistence"],
            "rootkit": ["root_access", "kernel_module", "hide_process", "syscall_hook"],
            "ddos": ["ddos", "dos_attack", "flood", "syn_flood", "http_flood", "ping_flood"],
            "data_exfiltration": ["exfil", "data_theft", "upload_data", "steal_data", "send_data"],
            "cryptominer": ["miner", "monero", "bitcoin", "cpu_usage", "mining_pool"]
        }
    })

def determine_malware_type(features):
    """Determine the likely type of malware based on detected patterns"""
    content_str = features['content_str'].lower()
    model = pickle.loads(st.session_state.model)
    malware_types = model["malware_types"]
    
    type_scores = {}
    
    # Count occurrences of indicators for each malware type
    for mal_type, indicators in malware_types.items():
        score = 0
        found_indicators = []
        
        for indicator in indicators:
            if indicator in content_str:
                score += 1
                found_indicators.append(indicator)
        
        if score > 0:
            type_scores[mal_type] = {
                "score": score,
                "indicators": found_indicators
            }
    
    # Also check behavior patterns
    # Network behavior suggests certain types
    if features['network_count'] > 2:
        if "botnet" in type_scores:
            type_scores["botnet"]["score"] += 2
        if "ddos" in type_scores:
            type_scores["ddos"]["score"] += 2
        if "data_exfiltration" in type_scores:
            type_scores["data_exfiltration"]["score"] += 2
    
    # System call behavior suggests certain types
    if features['system_call_count'] > 2:
        if "rootkit" in type_scores:
            type_scores["rootkit"]["score"] += 2
        if "backdoor" in type_scores:
            type_scores["backdoor"]["score"] += 2
    
    # Sort by score and return results
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return sorted_types

def extract_features(file_content, file_name):
    """Extract basic features from file content"""
    features = {}
    
    # File size
    features['file_size'] = len(file_content)
    
    # File hash
    features['md5'] = hashlib.md5(file_content).hexdigest()
    features['sha1'] = hashlib.sha1(file_content).hexdigest()
    
    # String analysis
    content_str = file_content.decode('utf-8', errors='ignore')
    features['content_str'] = content_str
    
    # Count suspicious strings
    model = pickle.loads(st.session_state.model)
    suspicious_count = 0
    found_suspicious = []
    
    for sus_string in model["malware_strings"]:
        if sus_string in content_str.lower():
            suspicious_count += 1
            found_suspicious.append(sus_string)
    
    features['suspicious_string_count'] = suspicious_count
    features['suspicious_strings'] = found_suspicious
    
    # Check for network functionality
    network_patterns = [r'http[s]?://', r'socket', r'connect\(', r'recv', r'send']
    network_matches = []
    for pattern in network_patterns:
        if re.search(pattern, content_str):
            network_matches.append(pattern)
    
    features['network_indicators'] = network_matches
    features['network_count'] = len(network_matches)
    
    # Check for system calls
    system_patterns = [r'system\(', r'exec\(', r'popen', r'spawn', r'fork\(']
    system_matches = []
    for pattern in system_patterns:
        if re.search(pattern, content_str):
            system_matches.append(pattern)
    
    features['system_indicators'] = system_matches
    features['system_call_count'] = len(system_matches)
    
    return features

def analyze_file(file_content, file_name):
    """Analyze file and determine if it's potentially malicious"""
    features = extract_features(file_content, file_name)
    
    # Simple heuristic-based detection
    risk_score = 0
    risk_factors = []
    
    # Suspicious strings check
    if features['suspicious_string_count'] > 0:
        risk_score += features['suspicious_string_count'] * 10
        risk_factors.append(f"Found {features['suspicious_string_count']} suspicious strings")
    
    # Network functionality check
    if features['network_count'] > 0:
        risk_score += features['network_count'] * 5
        risk_factors.append(f"Found {features['network_count']} network-related patterns")
    
    # System call check
    if features['system_call_count'] > 0:
        risk_score += features['system_call_count'] * 15
        risk_factors.append(f"Found {features['system_call_count']} system call patterns")
    
    # Determine risk level
    if risk_score > 50:
        risk_level = "High"
    elif risk_score > 20:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    # Determine potential malware types
    malware_types = determine_malware_type(features)
    
    return {
        "features": features,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "malware_types": malware_types
    }

def main():
    st.title("IoT Malware Detection Tool")
    
    st.write("""
    ## Upload an IoT application file for malware analysis
    This tool performs a basic static analysis to identify potential malicious indicators and malware types.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=["bin", "elf", "so", "ko", "py", "js", "exe"])
    
    if uploaded_file is not None:
        # Create a spinner to show the analysis is in progress
        with st.spinner("Analyzing file..."):
            # Read and analyze the file
            file_content = uploaded_file.read()
            file_name = uploaded_file.name
            result = analyze_file(file_content, file_name)
        
        # Display the analysis results
        st.subheader("Analysis Results")
        
        # File information
        st.write("### File Information")
        file_info = {
            "File Name": file_name,
            "File Size": f"{result['features']['file_size']} bytes",
            "MD5 Hash": result['features']['md5'],
            "SHA1 Hash": result['features']['sha1']
        }
        st.table(pd.DataFrame([file_info]))
        
        # Risk assessment
        st.write("### Risk Assessment")
        st.metric("Risk Score", result['risk_score'])
        
        # Color code the risk level
        if result['risk_level'] == "High":
            st.error(f"Risk Level: {result['risk_level']}")
        elif result['risk_level'] == "Medium":
            st.warning(f"Risk Level: {result['risk_level']}")
        else:
            st.success(f"Risk Level: {result['risk_level']}")
        
        # Malware type classification
        if result['malware_types']:
            st.write("### Potential Malware Classification")
            
            # Create a nice table for malware types
            types_data = []
            for mal_type, info in result['malware_types']:
                types_data.append({
                    "Malware Type": mal_type.replace('_', ' ').title(),
                    "Confidence Score": info["score"],
                    "Detected Indicators": ", ".join(info["indicators"]) if info["indicators"] else "Behavioral pattern"
                })
            
            if types_data:
                st.table(pd.DataFrame(types_data))
                
                # Show the primary type with highest confidence
                primary_type = result['malware_types'][0][0].replace('_', ' ').title()
                st.write(f"### Primary Classification: {primary_type}")
                
                # Display descriptions for the identified malware types
                st.write("#### About the detected malware types:")
                for mal_type, _ in result['malware_types']:
                    if mal_type == "botnet":
                        st.write("**Botnet**: Networks of compromised IoT devices controlled by attackers to perform distributed attacks.")
                    elif mal_type == "ransomware":
                        st.write("**Ransomware**: Malware that encrypts data and demands payment for decryption.")
                    elif mal_type == "backdoor":
                        st.write("**Backdoor**: Malware that creates hidden access points for attackers to maintain control.")
                    elif mal_type == "rootkit":
                        st.write("**Rootkit**: Sophisticated malware that hides its presence and provides privileged access.")
                    elif mal_type == "ddos":
                        st.write("**DDoS**: Software designed to flood targets with traffic, causing service disruption.")
                    elif mal_type == "data_exfiltration":
                        st.write("**Data Exfiltration**: Malware focused on stealing sensitive data from infected devices.")
                    elif mal_type == "cryptominer":
                        st.write("**Cryptominer**: Malware that uses device resources to mine cryptocurrency without consent.")
        
        # Risk factors
        if result['risk_factors']:
            st.write("### Risk Factors")
            for factor in result['risk_factors']:
                st.write(f"- {factor}")
            
            # Show suspicious strings if found
            if result['features']['suspicious_strings']:
                st.write("#### Suspicious Strings Detected")
                st.write(", ".join(result['features']['suspicious_strings']))
            
            # Show network indicators if found
            if result['features']['network_indicators']:
                st.write("#### Network Indicators")
                st.write(", ".join(result['features']['network_indicators']))
            
            # Show system call indicators if found
            if result['features']['system_indicators']:
                st.write("#### System Call Indicators")
                st.write(", ".join(result['features']['system_indicators']))
        else:
            st.success("No significant risk factors detected")
        
        # Display recommendation
        st.subheader("Recommendation")
        if result['risk_level'] == "High":
            st.error("This file contains multiple indicators of potentially malicious behavior. We recommend not installing or using this application without further analysis.")
        elif result['risk_level'] == "Medium":
            st.warning("This file contains some suspicious indicators. Proceed with caution and consider additional verification.")
        else:
            st.success("This file appears to be relatively safe based on our analysis, but no detection method is 100% accurate.")
    
    st.sidebar.title("About")
    st.sidebar.info("""
    This IoT Malware Detection Tool performs basic static analysis to identify potential security risks in IoT applications.
    
    **Features:**
    - File hash generation and verification
    - Suspicious string detection
    - Network functionality analysis
    - System call detection
    - Malware type classification
    
    **Detectable Malware Types:**
    - Botnets
    - Ransomware
    - Backdoors
    - Rootkits
    - DDoS tools
    - Data exfiltration
    - Cryptominers
    
    **Note:** This is a simplified demonstration tool and should not replace comprehensive security auditing.
    """)

if __name__ == "__main__":
    main()
