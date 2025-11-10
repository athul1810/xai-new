"""
Create a comprehensive synthetic dataset for substance-abuse-risk detection.
"""

import pandas as pd
import random
from pathlib import Path

def create_synthetic_dataset(output_path='data/raw.csv', num_samples=500):
    """
    Create a synthetic dataset with diverse examples of high-risk and low-risk text.
    
    Args:
        output_path: Path to save the CSV file
        num_samples: Total number of samples (will be split roughly 50/50)
    """
    
    # High-risk patterns (label=1)
    high_risk_patterns = [
        # Substance use mentions
        "I've been drinking heavily every night and can't stop.",
        "Struggling with substance use and need help.",
        "Can't function without my daily dose.",
        "Using drugs to cope with stress and anxiety.",
        "Using drugs to cope with stress.",
        "I use drugs to cope with stress.",
        "I'm using drugs to cope with stress.",
        "Drugs help me cope with stress.",
        "I rely on drugs to cope with stress.",
        "I need help with my addiction.",
        "Been using substances to numb the pain.",
        "Can't go a day without drinking.",
        "My drug use is getting out of control.",
        "I'm addicted and don't know what to do.",
        "Using pills to sleep every night.",
        "Drinking to forget my problems.",
        "I've been using cocaine regularly.",
        "My alcohol consumption is concerning.",
        "Can't stop using marijuana daily.",
        "Prescription drug abuse is ruining my life.",
        "I'm high all the time to escape reality.",
        "Substance abuse is destroying my relationships.",
        "I need my fix every few hours.",
        "Drinking alone every evening.",
        "Using drugs to feel normal.",
        
        # Mental health + substance
        "Depressed and using alcohol to cope.",
        "Anxiety makes me drink more.",
        "I use substances when I'm stressed.",
        "Can't handle emotions without drugs.",
        "Drinking because I'm so depressed.",
        "Using pills to manage my anxiety.",
        "Substance use is my only escape.",
        "I drink to numb emotional pain.",
        "Drugs help me forget my problems.",
        "Can't deal with life without substances.",
        
        # Risky behaviors
        "I've been blacking out from drinking.",
        "Drove while under the influence.",
        "Lost my job due to substance use.",
        "Family is worried about my drinking.",
        "I've been hiding my drug use.",
        "Spending all my money on substances.",
        "Neglecting responsibilities because of drugs.",
        "I've overdosed before.",
        "My health is deteriorating from substance use.",
        "Can't remember what happened last night.",
        
        # Desperation/help-seeking
        "I need help but don't know where to turn.",
        "My addiction is taking over my life.",
        "I want to stop but can't.",
        "Substance use is my biggest problem.",
        "I'm scared of my own behavior.",
        "Help me, I'm losing control.",
        "I can't stop using on my own.",
        "My substance use is escalating.",
        "I need professional help for addiction.",
        "I'm in denial about my problem.",
    ]
    
    # Low-risk patterns (label=0)
    low_risk_patterns = [
        # Positive activities
        "Feeling great today! Had a productive morning.",
        "Just finished a workout, feeling energized!",
        "Had a wonderful day with family and friends.",
        "Enjoying a healthy lifestyle and good habits.",
        "Feeling grateful and happy today.",
        "Had a great time at the park.",
        "Enjoyed a nice meal with loved ones.",
        "Feeling motivated and positive.",
        "Had a productive day at work.",
        "Enjoying my hobbies and interests.",
        "Feeling content and peaceful.",
        "Had a relaxing weekend.",
        "Feeling healthy and strong.",
        "Enjoying good company and conversation.",
        "Feeling optimistic about the future.",
        
        # Normal daily activities
        "Just finished reading a good book.",
        "Went for a walk in nature.",
        "Cooked a healthy meal today.",
        "Spent time with my pets.",
        "Cleaned the house and feel accomplished.",
        "Watched a movie with friends.",
        "Listened to music and relaxed.",
        "Did some gardening today.",
        "Went shopping for groceries.",
        "Attended a community event.",
        "Helped a neighbor with something.",
        "Learned something new today.",
        "Made progress on my goals.",
        "Had a good conversation with someone.",
        "Felt proud of my achievements.",
        
        # Healthy coping
        "Meditation helps me stay calm.",
        "Exercise is my stress relief.",
        "Talking to friends helps me feel better.",
        "I practice mindfulness daily.",
        "Journaling helps me process emotions.",
        "I have healthy ways to cope.",
        "Therapy has been helpful for me.",
        "I take care of my mental health.",
        "I have a good support system.",
        "I practice self-care regularly.",
        
        # Positive mental health
        "Feeling balanced and centered.",
        "My mental health is stable.",
        "I have good coping strategies.",
        "Feeling resilient and strong.",
        "I'm managing stress well.",
        "Feeling emotionally stable.",
        "I have healthy relationships.",
        "Feeling confident and capable.",
        "I'm making good life choices.",
        "Feeling hopeful and positive.",
        
        # General positive statements
        "Life is good right now.",
        "I'm grateful for what I have.",
        "Feeling blessed and fortunate.",
        "Things are going well for me.",
        "I'm happy with my progress.",
        "Feeling satisfied with life.",
        "I have a lot to be thankful for.",
        "Feeling peaceful and content.",
        "I'm in a good place mentally.",
        "Life feels meaningful and purposeful.",
    ]
    
    # Create dataset with more diversity
    data = []
    
    # Generate high-risk samples
    num_high_risk = num_samples // 2
    for _ in range(num_high_risk):
        # Use exact patterns more often to ensure quality
        if random.random() < 0.8:
            text = random.choice(high_risk_patterns)
        else:
            # Create slight variations
            base = random.choice(high_risk_patterns)
            variations = [
                base,
                base.lower(),
                base.capitalize(),
                base + " It's really hard.",
                "Lately, " + base.lower(),
                base + " I don't know what to do.",
            ]
            text = random.choice(variations)
        data.append({'text': text, 'label': 1})
    
    # Generate low-risk samples - ensure they're clearly positive
    num_low_risk = num_samples - num_high_risk
    for _ in range(num_low_risk):
        if random.random() < 0.8:
            text = random.choice(low_risk_patterns)
        else:
            base = random.choice(low_risk_patterns)
            variations = [
                base,
                base.lower(),
                base.capitalize(),
                base + " Feeling good!",
                "Today, " + base.lower(),
                base + " Life is good.",
            ]
            text = random.choice(variations)
        data.append({'text': text, 'label': 0})
    
    # Shuffle the data
    random.shuffle(data)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created synthetic dataset: {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   High risk (label=1): {len(df[df['label']==1])}")
    print(f"   Low risk (label=0): {len(df[df['label']==0])}")
    print(f"\nSample high-risk texts:")
    for text in df[df['label']==1]['text'].head(3).tolist():
        print(f"   - {text}")
    print(f"\nSample low-risk texts:")
    for text in df[df['label']==0]['text'].head(3).tolist():
        print(f"   - {text}")
    
    return df

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create dataset with 500 samples
    create_synthetic_dataset(num_samples=500)

