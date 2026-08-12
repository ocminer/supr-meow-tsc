#!/usr/bin/env python3
"""
Simple script to run blockchain AI inference requests with randomized parameters.
No external dependencies required - uses only Python standard library.
"""

import json
import urllib.request
import urllib.error
import random
import string
import base64
import argparse
import time
from typing import List, Dict, Any

# Prompt bank - 100+ prompts organized in groups of 4
PROMPT_BANK = [
    # Group 1
    [
        "List the benefits of dual proof of work scheme that allow to secure a blockchain while running AI inference",
        "Explain how blockchain technology can enhance AI model verification and trust",
        "What are the main challenges in combining blockchain with AI inference systems?",
        "Describe the potential energy efficiency improvements in blockchain-AI hybrid systems"
    ],
    # Group 2
    [
        "How can decentralized AI inference improve data privacy and security?",
        "What role does consensus play in distributed AI computation networks?",
        "Explain the concept of federated learning in blockchain environments",
        "List the advantages of tokenizing AI compute resources on blockchain"
    ],
    # Group 3
    [
        "Describe how smart contracts can automate AI model deployment and updates",
        "What are the implications of immutable AI decision logs on blockchain?",
        "How can blockchain prevent AI model tampering and ensure integrity?",
        "Explain the benefits of decentralized AI marketplaces"
    ],
    # Group 4
    [
        "What are the scalability considerations for blockchain-based AI systems?",
        "How can zero-knowledge proofs enhance AI privacy on blockchain?",
        "Describe the role of oracles in connecting AI systems with blockchain",
        "List the potential use cases for AI-powered blockchain analytics"
    ],
    # Group 5
    [
        "How can blockchain incentivize contribution to distributed AI training?",
        "What are the benefits of storing AI model hashes on blockchain?",
        "Explain how blockchain can enable AI model versioning and rollback",
        "Describe the concept of AI governance through blockchain voting"
    ],
    # Group 6
    [
        "What are the advantages of peer-to-peer AI inference networks?",
        "How can blockchain ensure fair compensation for AI compute providers?",
        "Explain the role of reputation systems in decentralized AI networks",
        "List the security benefits of distributed AI model storage"
    ],
    # Group 7
    [
        "How can blockchain enable transparent AI decision-making processes?",
        "What are the benefits of cryptographic proofs in AI computations?",
        "Describe how blockchain can facilitate AI model sharing and collaboration",
        "Explain the concept of AI-as-a-Service on blockchain platforms"
    ],
    # Group 8
    [
        "What are the advantages of using blockchain for AI data provenance?",
        "How can smart contracts enforce AI ethical guidelines?",
        "Describe the potential of blockchain in AI model certification",
        "List the benefits of decentralized AI training data management"
    ],
    # Group 9
    [
        "How can blockchain prevent single points of failure in AI systems?",
        "What role does tokenomics play in incentivizing AI development?",
        "Explain how blockchain can enable AI model micropayments",
        "Describe the benefits of consensus mechanisms for AI validation"
    ],
    # Group 10
    [
        "What are the advantages of blockchain-based AI audit trails?",
        "How can decentralized identity enhance AI personalization?",
        "Explain the concept of AI model staking on blockchain",
        "List the benefits of cross-chain AI interoperability"
    ],
    # Group 11
    [
        "How can blockchain enable secure multi-party AI computation?",
        "What are the benefits of using blockchain for AI dataset licensing?",
        "Describe how blockchain can facilitate AI research collaboration",
        "Explain the role of decentralized storage in AI model distribution"
    ],
    # Group 12
    [
        "What are the advantages of blockchain-based AI performance benchmarking?",
        "How can smart contracts automate AI model quality assurance?",
        "Describe the potential of blockchain in AI bias detection and mitigation",
        "List the benefits of decentralized AI model registries"
    ],
    # Group 13
    [
        "How can blockchain enable verifiable AI training processes?",
        "What are the benefits of using blockchain for AI compute scheduling?",
        "Explain how blockchain can facilitate AI model monetization",
        "Describe the concept of decentralized AI orchestration"
    ],
    # Group 14
    [
        "What are the advantages of blockchain-based AI resource allocation?",
        "How can blockchain ensure AI model reproducibility?",
        "Explain the role of consensus in AI result verification",
        "List the benefits of blockchain for AI intellectual property protection"
    ],
    # Group 15
    [
        "How can blockchain enable trustless AI service agreements?",
        "What are the benefits of decentralized AI model evaluation?",
        "Describe how blockchain can facilitate AI knowledge graphs",
        "Explain the concept of AI mining on blockchain networks"
    ],
    # Group 16
    [
        "What are the advantages of blockchain-based AI data marketplaces?",
        "How can smart contracts enforce AI service level agreements?",
        "Describe the potential of blockchain in AI explainability",
        "List the benefits of decentralized AI inference caching"
    ],
    # Group 17
    [
        "How can blockchain enable secure AI model updates?",
        "What are the benefits of using blockchain for AI compliance tracking?",
        "Explain how blockchain can facilitate AI model composition",
        "Describe the role of decentralized governance in AI systems"
    ],
    # Group 18
    [
        "What are the advantages of blockchain-based AI quality metrics?",
        "How can blockchain prevent AI model poisoning attacks?",
        "Explain the concept of AI compute proof-of-work",
        "List the benefits of blockchain for AI data integrity"
    ],
    # Group 19
    [
        "How can blockchain enable fair AI resource distribution?",
        "What are the benefits of decentralized AI model discovery?",
        "Describe how blockchain can facilitate AI error reporting",
        "Explain the role of cryptoeconomics in AI networks"
    ],
    # Group 20
    [
        "What are the advantages of blockchain-based AI access control?",
        "How can smart contracts manage AI model lifecycles?",
        "Describe the potential of blockchain in AI cost optimization",
        "List the benefits of decentralized AI monitoring systems"
    ],
    # Group 21
    [
        "How can blockchain enable verifiable AI benchmarks?",
        "What are the benefits of using blockchain for AI data validation?",
        "Explain how blockchain can facilitate AI model aggregation",
        "Describe the concept of AI consensus algorithms"
    ],
    # Group 22
    [
        "What are the advantages of blockchain-based AI reputation systems?",
        "How can blockchain ensure AI service availability?",
        "Explain the role of tokens in AI compute markets",
        "List the benefits of decentralized AI debugging"
    ],
    # Group 23
    [
        "How can blockchain enable secure AI model sharing?",
        "What are the benefits of decentralized AI testing frameworks?",
        "Describe how blockchain can facilitate AI performance tracking",
        "Explain the concept of AI proof-of-stake systems"
    ],
    # Group 24
    [
        "What are the advantages of blockchain-based AI data cleaning?",
        "How can smart contracts automate AI model selection?",
        "Describe the potential of blockchain in AI feature engineering",
        "List the benefits of decentralized AI experimentation"
    ],
    # Group 25
    [
        "How can blockchain enable transparent AI funding mechanisms?",
        "What are the benefits of using blockchain for AI model comparison?",
        "Explain how blockchain can facilitate AI knowledge transfer",
        "Describe the role of decentralized AI optimization"
    ],
    # Group 26 (final group to exceed 100 prompts)
    [
        "What are the advantages of blockchain-based AI collaboration tools?",
        "How can blockchain prevent unauthorized AI model usage?",
        "Explain the concept of AI computation verification on blockchain",
        "List the benefits of decentralized AI deployment strategies"
    ]
]

def generate_random_hex(length: int) -> str:
    """Generate a random hexadecimal string of specified length."""
    return ''.join(random.choices('0123456789abcdef', k=length))

def generate_random_block_hash() -> str:
    """Generate a random 64-character block hash."""
    return generate_random_hex(64)

def generate_random_vdf() -> str:
    """Generate a random VDF string (similar format to the example)."""
    # Generate random hex data similar to the VDF format in the example
    parts = []
    # First part: 2 chars
    parts.append(generate_random_hex(2))
    # Second part: 256 chars
    parts.append(generate_random_hex(256))
    # Third part: 8 chars
    parts.append(generate_random_hex(8))
    # Fourth part: 256 chars
    parts.append(generate_random_hex(256))
    
    return ''.join(parts)

def generate_random_header_prefix() -> str:
    """Generate a random base64-encoded header prefix."""
    # Generate random bytes and encode to base64
    random_bytes = bytes(random.getrandbits(8) for _ in range(64))
    return base64.b64encode(random_bytes).decode('utf-8')

def get_prompt_batch(batch_index: int) -> List[str]:
    """Get a batch of 4 prompts from the prompt bank."""
    return PROMPT_BANK[batch_index % len(PROMPT_BANK)]

def make_request(
    model: str,
    prompts: List[str],
    top_k: int,
    top_p: float,
    block_hash: str,
    vdf: str,
    header_prefix: str,
    url: str = "http://localhost:8000/v1/completions"
) -> Dict[str, Any]:
    """Make a single request to the API."""
    
    # Prepare the request data
    data = {
        "model": model,
        "prompt": prompts,
        "max_tokens": 256,
        "temperature": 1.0,
        "top_k": top_k,
        "top_p": top_p,
        "extra_sampling_params": {
            "pow": {
                "block_hash": block_hash,
                "vdf": vdf,
                "tick": 44,
                "target": "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "header_prefix": header_prefix
            }
        }
    }
    
    # Prepare the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dev-secret"
    }
    
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method='POST'
    )
    
    try:
        # Make the request
        with urllib.request.urlopen(request) as response:
            return {
                "status": response.status,
                "data": json.loads(response.read().decode('utf-8'))
            }
    except urllib.error.HTTPError as e:
        return {
            "status": e.code,
            "error": e.reason,
            "data": e.read().decode('utf-8')
        }
    except Exception as e:
        return {
            "status": -1,
            "error": str(e)
        }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run blockchain AI inference requests with randomized parameters'
    )
    parser.add_argument('--top_k', type=int, default=30, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling parameter')
    parser.add_argument('--model', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 
                        help='Model to use for inference')
    parser.add_argument('-n', '--num_requests', type=int, default=10, 
                        help='Number of requests to make')
    parser.add_argument('--delay', type=float, default=1.0, 
                        help='Delay between requests in seconds')
    parser.add_argument('--url', type=str, default='http://localhost:8000/v1/completions',
                        help='API endpoint URL')
    
    args = parser.parse_args()
    
    print(f"Starting {args.num_requests} requests with the following parameters:")
    print(f"  Model: {args.model}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Delay between requests: {args.delay}s")
    print(f"  URL: {args.url}")
    print("-" * 80)
    
    # Track statistics
    successful_requests = 0
    failed_requests = 0
    
    # Make N requests
    for i in range(args.num_requests):
        print(f"\nRequest {i + 1}/{args.num_requests}")
        
        # Generate random parameters
        block_hash = generate_random_block_hash()
        vdf = generate_random_vdf()
        header_prefix = generate_random_header_prefix()
        
        # Get prompt batch (cycles through the prompt bank)
        prompt_batch_index = i // 4  # New batch every 4 requests
        prompts = get_prompt_batch(prompt_batch_index)
        
        print(f"  Block hash: {block_hash[:16]}...")
        print(f"  VDF: {vdf[:32]}...")
        print(f"  Header prefix: {header_prefix[:20]}...")
        print(f"  Using prompt batch {prompt_batch_index + 1}")
        
        # Make the request
        result = make_request(
            model=args.model,
            prompts=prompts,
            top_k=args.top_k,
            top_p=args.top_p,
            block_hash=block_hash,
            vdf=vdf,
            header_prefix=header_prefix,
            url=args.url
        )
        
        # Process the result
        if result["status"] == 200:
            successful_requests += 1
            print(f"  Status: SUCCESS")
            if "data" in result and isinstance(result["data"], dict):
                # Print a sample of the response if available
                if "choices" in result["data"] and result["data"]["choices"]:
                    text = result["data"]["choices"][0].get("text", "")
                    if text:
                        preview = text[:100] + "..." if len(text) > 100 else text
                        print(f"  Response preview: {preview}")
        else:
            failed_requests += 1
            print(f"  Status: FAILED ({result.get('status', 'Unknown')})")
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        # Wait before next request (except for the last one)
        if i < args.num_requests - 1:
            time.sleep(args.delay)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Total requests: {args.num_requests}")
    print(f"Successful: {successful_requests}")
    print(f"Failed: {failed_requests}")
    print(f"Success rate: {successful_requests / args.num_requests * 100:.1f}%")

if __name__ == "__main__":
    main()