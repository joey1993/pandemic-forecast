# pandemic-forecast

## Step1 Data Preprocessing
### Build NER Dataset
`bash s0_tweets_preparation.sh`

### Run Named Entity Recognition
`bash s1_named_entity_recognition.sh`

### Extract Significant Entities and Events
`bash s2_extract_entity_hashtag.sh`

### Run Relation Extraction
`bash s3_relation_extraction.sh`

### Build Nodes and Edges for Time Series Prediction Model
`bash s4_node_edge_construction.sh`

## Step2 Run Graph-based Pandemic Forecast
`bash s6_time_series_prediction.sh`
