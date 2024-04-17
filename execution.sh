arch=$1
opt=$2
 
# Input size matters
echo "codegen2"
python3.11 codegen2_script.py --base_model="Salesforce/codegen2-16B" --arch="${arch}" --opt="${opt}"