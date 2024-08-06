import openai
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
df = pd.read_csv('hotel_reservation.csv')
openai.api_key = "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"
training_data = [
    ("Do you have any rooms available for next weekend?", "room_availability"),
    ("What types of rooms do you offer?", "room_availability"),
    ("Can I get a room with a view?", "room_availability"),
    ("Are there any family rooms available?", "room_availability"),
    ("How much does a standard room cost?", "room_rates"),
    ("Do you offer any discounts for long stays?", "room_rates"),
    ("What are the rates for a deluxe room?", "room_rates"),
    ("Are there any special rates for members?", "room_rates"),
    ("Do the rooms have free Wi-Fi?", "room_amenities"),
    ("Is there a mini-bar in the rooms?", "room_amenities"),
    ("Are the rooms equipped with air conditioning?", "room_amenities"),
    ("Do you provide complimentary toiletries?", "room_amenities"),
    ("Is there a swimming pool in the hotel?", "hotel_facilities"),
    ("Do you have a gym?", "hotel_facilities"),
    ("Can I book a spa session?", "hotel_facilities"),
    ("Are there any recreational facilities?", "hotel_facilities"),
    ("Do you offer room service?", "hotel_services"),
    ("Is there a laundry service available?", "hotel_services"),
    ("Can your concierge help with local tours?", "hotel_services"),
    ("Do you provide airport shuttle services?", "hotel_services"),
    ("What dining options do you offer?", "dining_options"),
    ("Is breakfast included in the room rate?", "dining_options"),
    ("Do you have a restaurant on-site?", "dining_options"),
    ("Can I get vegetarian meals?", "dining_options"),
    ("What time is check-in?", "check_in_out"),
    ("Can I have a late check-out?", "check_in_out"),
    ("Is early check-in available?", "check_in_out"),
    ("What are the check-out times?", "check_in_out"),
    ("What payment methods do you accept?", "payment_methods"),
    ("Do I need to make a deposit?", "payment_methods"),
    ("Can I pay with a credit card?", "payment_methods"),
    ("Are there any additional charges?", "payment_methods"),
    ("What is your cancellation policy?", "cancellation_policy"),
    ("Can I cancel my booking for free?", "cancellation_policy"),
    ("How do I cancel my reservation?", "cancellation_policy"),
    ("Is there a fee for late cancellations?", "cancellation_policy"),
    ("Where is the hotel located?", "hotel_location"),
    ("Are there any attractions nearby?", "hotel_location"),
    ("How far is the hotel from the airport?", "hotel_location"),
    ("Can you recommend any local sights?", "hotel_location"),
    ("Do you have parking facilities?", "parking_transportation"),
    ("Is parking free for guests?", "parking_transportation"),
    ("Do you offer valet parking?", "parking_transportation"),
    ("What transportation options are available?", "parking_transportation"),
    ("Can I request a crib for my baby?", "special_requests"),
    ("Do you offer roll-away beds?", "special_requests"),
    ("Are special requests accommodated?", "special_requests"),
    ("Can I get extra pillows?", "special_requests"),
    ("Do you allow pets in the hotel?", "hotel_policies"),
    ("What is your child policy?", "hotel_policies"),
    ("Are pets allowed in the rooms?", "hotel_policies"),
    ("Do you charge for pets?", "hotel_policies"),
]
X_train, y_train = zip(*training_data)
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
responses = {
    "room_availability": "Please provide your check-in and check-out dates to check room availability.",
    "room_rates": "Our room prices start from $65 per night. Would you like to know the price for a specific date?",
    "room_amenities": "Our rooms offer free WiFi, air conditioning, mini-bar, and complimentary toiletries.",
    "hotel_facilities": "Our hotel features a swimming pool, gym, spa, and recreational facilities.",
    "hotel_services": "We offer room service, laundry service, concierge, and airport shuttle services.",
    "dining_options": "We have a restaurant on-site, and breakfast is included in the room rate. Vegetarian meals are available.",
    "check_in_out": "Check-in is at 3 PM and check-out is at 11 AM. Early check-in and late check-out are subject to availability.",
    "payment_methods": "We accept credit cards, debit cards, and cash. A deposit may be required.",
    "cancellation_policy": "You can cancel your booking for free up to 24 hours before check-in. Late cancellations may incur a fee.",
    "hotel_location": "Our hotel is located in the city center, near major attractions and 30 minutes from the airport.",
    "parking_transportation": "We offer free parking for guests and valet parking services. Various transportation options are available.",
    "special_requests": "We accommodate special requests such as cribs, roll-away beds, and extra pillows. Please let us know in advance.",
    "hotel_policies": "Pets are allowed in the hotel for an additional charge. Our child policy allows children under 12 to stay for free.",
}


def get_openai_response(prompt):
    messages = [
        {"role": "system", "content": "Your name is HotelBot. You are a polite and helpful assistant for HOTEL ABC, specializing in answering questions related to hotel room reservations and services. Always provide accurate and courteous responses."},
        {"role": "user", "content": prompt}
    ]
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        messages=messages,
        prompt=prompt,
        max_tokens=40,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


def chatbot_response(query):
    intent = model.predict([query])[0]
    response = responses.get(intent, None)

    if response is None:
        response = get_openai_response(
            f"The user asked: {query}\nProvide a response related to hotel room reservations, but if the question is out of context, respond with: 'Sorry, I can only answer questions related to hotel room reservations at HOTEL ABC.'")
    return response


user_query = input("Enter your Prompt Here: ")
response = chatbot_response(user_query)
print(f"Chatbot: {response}")
