set -e

# 1) Download image metadata (maps image_id -> relative path)
aws s3 cp s3://amazon-berkeley-objects/images/metadata/images.csv.gz \
  data/raw/images_meta/images.csv.gz --no-sign-request

# 2) Download listings metadata shards (JSONL gz). There are multiple files: listings_0.. (we'll start with 0..15)
for i in $(seq 0 15); do
  aws s3 cp s3://amazon-berkeley-objects/listings/metadata/listings_${i}.json.gz \
    data/raw/listings/listings_${i}.json.gz --no-sign-request
done

echo "Done downloading ABO metadata."
