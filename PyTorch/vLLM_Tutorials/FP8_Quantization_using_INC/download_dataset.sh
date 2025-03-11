wget https://rclone.org/install.sh
chmod a+x ./install.sh
./install.sh

rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca /root/open_orca -P

gzip -c -d /root/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz > /root/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl