---
title: "Activity 5 Reflection - Constituent Services Hub"
type: reflection
version: "1.0.0"
---

# Activity 5 Reflection

Answer each question in 3-5 sentences. Thoughtful, specific responses earn full credit.

## 1. PII Redaction in Practice

What PII categories did the Azure AI Language service detect in the Memphis 311 complaints? Did you notice any false positives (non-PII flagged as sensitive) or false negatives (PII that was missed)? How might PII detection requirements differ between a government agency processing citizen complaints and a commercial customer service system?

Azure AI Language Service detected the following categories: Person, USSocialSecurityNumber, Address, PhoneNumber, PersonType, Email, DateTime, Organization. It gave false positives for a case number '311-2025-044' as a phone number, a word outside of a street name 'calle Union Avenue', and recognized 'now' but not 'two weeks' as a datetime in 'two weeks now'. Government entities have a legal obligation to redact pii and a commercial systems focus on data protection during collection and consumer rights to access. 

## 2. Sentiment as a Routing Signal

How could sentiment analysis be used to prioritize Memphis 311 complaints — for example, routing highly negative complaints to senior staff? What are the risks of relying solely on sentiment for prioritization (consider complaints that are neutral in tone but urgent in nature)? How would you combine sentiment with other signals for a more robust routing system?

Flag high-risk, negative messages for immediate intervention, route complex issues to senior agents, and identifies which complaints pose the highest risk. Relying solely on sentiment for prioritization risks missing urgent, high-impact issues that are phrased neutrally, confuse volume with urgency, fail to detect sarcasm, and suffer from domain-specific inaccuracies. Route a high-value customer with negative sentiment directly to a specialized retention team or combine negative sentiment with urgent, high-value keywords to elevate the ticket's priority. 

## 3. Multilingual Challenges

How accurately did the Language service detect the language of short versus long text samples? What challenges might arise with code-switching (mixing languages in a single message), which is common in multilingual communities? How would you handle a complaint where language detection confidence is low?

Azure AI Language Service detected false positives for a word outside of a street name, 'calle Union Avenue' with a confidence score of .84, in a spanish sentence. Code-switching in Azure AI Language Services poses challenges in accuracy, language identification, and sentiment analysis. Key issues include reduced accuracy in custom models, potential failures in speech-to-text with mixed input, and difficulty distinguishing between languages with similar alphabets. To handle low-confidence language detection in Azure AI Language Service, improve accuracy by providing more context, using countryHint parameters, or upgrading to custom classification for specific domain jargon. If confidence remains low, route the input to a human agent or assume a default language, as low confidence often results from mixed languages or very short text.

## 4. CLU vs. Keyword Matching

Compare the CLU model's intent classification with the keyword-based fallback. In what scenarios did CLU perform better, and where did keyword matching suffice? What are the trade-offs of training and maintaining a custom CLU model versus using a simpler rule-based approach for a city's 311 system? *(If CLU was not configured in your environment, discuss how you would expect it to differ based on the training data in `data/intent_examples.json`.)*

The CLU model's intent classification uses Natural Language Understanding models to recognize semantic intent. Keyword-Based Fallback uses deterministic rules to match specific trigger phrases or words. CLU performed better in contextual understanding, multilingual support, semantic complexity, and entity extraction. Keyword matching sufficed when intent was explicit and unambiguous, text contained clear trigger words without conflicting context, and only 3 of 10 complaints had keyword-matching agreement with CLU. CLU requires training data, deployment infrastructure, and ongoing maintenance but handles edge cases, multilingual input, and provides rich entity extraction. Keyword matching is simpler and faster but misses 70% of nuanced intents in this dataset, making it insufficient for a city's diverse 311 system where citizens phrase complaints in varied, contextual ways.

## 5. Pipeline Design

Which step in your NLP pipeline was most critical to get right first, and why? If you were deploying this pipeline for production use handling thousands of complaints daily, what monitoring, fallback mechanisms, or error handling would you add? How would you measure pipeline health over time?

PII redaction was the most critical step to implement first because once sensitive data leaks into logs or downstream systems, you have an irreversible compliance violation. For production deployment handling thousands of daily complaints, you'd need add real-time monitoring of success rates and latency per step, circuit breakers with retry logic for Azure API failures with the existing the keyword fallback to maintain availability during CLU outages. Error handling should allow partial pipeline completion so a failure in sentiment analysis doesn't prevent translation and intent recognition from completing. Pipeline health would be measured by tracking per-step success rates, and stablish baselines for confidence scores and entities-per-complaint to detect model drift.
