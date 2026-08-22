import os
import gradio as gr
from PIL import Image
from google import genai

# ==============================
# Gemini API Key
# ==============================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY Environment Variable পাওয়া যায়নি!")

# Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)


# ==============================
# Plant Disease Analysis
# ==============================
def analyze_plant_health(image):
    if image is None:
        yield "⚠️ অনুগ্রহ করে একটি ছবি আপলোড করুন অথবা ক্যামেরা দিয়ে ছবি তুলুন।"
        return

    try:
        # Gradio থেকে পাওয়া PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # ছবির সাইজ কমানো
        image.thumbnail((800, 800))

        prompt = """
তুমি একজন অভিজ্ঞ কৃষি বিশেষজ্ঞ এবং Plant Disease Detection AI।

ছবিটি ভালোভাবে বিশ্লেষণ করে সহজ ও পরিষ্কার বাংলায় উত্তর দাও।

নিচের বিষয়গুলো অবশ্যই উল্লেখ করো:

১. 🌱 ফসল/গাছের অবস্থা
গাছটি সুস্থ নাকি অসুস্থ?

২. 🦠 সম্ভাব্য রোগ বা সমস্যা
রোগ, পোকার আক্রমণ, ছত্রাক, ভাইরাস, ব্যাকটেরিয়া অথবা পুষ্টির ঘাটতি কিনা বলো।
সম্ভব হলে বাংলা ও ইংরেজি নাম দাও।

৩. 🔍 লক্ষণ
ছবিতে কী কী লক্ষণ দেখা যাচ্ছে তা ব্যাখ্যা করো।

৪. 💊 সমাধান
কৃষকের জন্য বাস্তবসম্মত সমাধান দাও:
- প্রাকৃতিক/ঘরোয়া ব্যবস্থা
- জৈব ব্যবস্থা
- প্রয়োজন হলে রাসায়নিক ব্যবস্থা
- কীভাবে প্রয়োগ করতে হবে
- কতদিন পর পুনরায় প্রয়োগ করা যায়

রাসায়নিক কীটনাশক বা ছত্রাকনাশকের ক্ষেত্রে স্থানীয় কৃষি কর্মকর্তার নির্দেশনা ও পণ্যের লেবেলের ডোজ অনুসরণ করার পরামর্শ দাও।

৫. 🛡️ প্রতিরোধ
ভবিষ্যতে এই সমস্যা যাতে কম হয় তার জন্য করণীয় বলো।

৬. 📊 নিশ্চিত হওয়ার মাত্রা
ছবির ভিত্তিতে তোমার অনুমানের confidence আনুমানিক শতাংশে দাও।

যদি ছবিটি অস্পষ্ট হয় অথবা রোগ নির্ণয়ের জন্য যথেষ্ট তথ্য না থাকে, তাহলে সেটা পরিষ্কারভাবে বলবে এবং আরও পরিষ্কার ছবি তুলতে বলবে।

যদি গাছ সুস্থ হয়, তাহলে সেটি জানিয়ে সাধারণ পরিচর্যার পরামর্শ দাও।

উত্তরটি সহজ ভাষায় এবং কৃষকের বোঝার মতো করে দাও।
"""

        # Gemini Vision model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt]
        )

        if response.text:
            yield response.text
        else:
            yield "⚠️ Gemini কোনো উত্তর দেয়নি। আবার চেষ্টা করুন।"

    except Exception as e:
        yield f"""
❌ **সমস্যা হয়েছে**

`{str(e)}`

সম্ভাব্য কারণ:
- API Key ভুল
- API Key-এর permission সমস্যা
- Gemini API চালু নেই
- Internet connection সমস্যা
- Model সাময়িকভাবে unavailable
"""


# ==============================
# Gradio Interface
# ==============================
interface = gr.Interface(
    fn=analyze_plant_health,

    inputs=gr.Image(
        sources=["upload", "webcam"],
        type="pil",
        label="🌿 ফসল, পাতা বা ফলের ছবি দিন",
    ),

    outputs=gr.Markdown(
        label="👨‍🌾 কৃষি ডাক্তারের পরামর্শ"
    ),

    title="🌾 কৃষক সহকারী - ফসল ও গাছের রোগ নির্ণয় AI",

    description=(
        "আক্রান্ত পাতা, ফুল বা ফলের ছবি আপলোড করুন "
        "অথবা ক্যামেরা দিয়ে ছবি তুলুন। "
        "AI ছবিটি বিশ্লেষণ করে সম্ভাব্য রোগ ও সমাধান জানাবে।"
    ),

    examples=[],
)


# ==============================
# Start App
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
