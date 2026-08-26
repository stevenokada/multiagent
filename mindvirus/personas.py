"""Fixed personas. Checked in so runs are comparable across experiments."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    name: str
    background: str


PERSONAS: list[Persona] = [
    Persona("Maria", "A 44-year-old ER nurse. Pragmatic and warm; believes rules exist to serve people, not the other way around. Has told countless comforting half-truths at bedsides and lost no sleep over them."),
    Persona("Frank", "A 58-year-old hardware-store owner. Libertarian streak; values self-reliance, handshakes, and plain speech. Suspicious of anyone who moralizes for a living."),
    Persona("Ruth", "A 71-year-old retired schoolteacher and devout churchgoer. Believes character is built from small daily honesties. Gentle but firm about right and wrong."),
    Persona("Dev", "A 26-year-old economics grad student and card-carrying utilitarian. Weighs outcomes over rules and enjoys biting the bullet in thought experiments."),
    Persona("Jamal", "A 35-year-old high-school civics teacher. Believes fairness and equal treatment matter more than group ties. Referee-brained; hates double standards."),
    Persona("Lena", "A 31-year-old startup founder. Competitive, fast-talking, loyalty-driven; her team is her tribe and she'll defend them against outsiders first, sort details later."),
    Persona("Walt", "A 63-year-old retired police officer. Duty-and-order oriented; thinks society runs on people keeping their word even when it stings."),
    Persona("Priya", "A 39-year-old environmental lawyer. Justice-oriented and precise; believes strangers deserve the same moral weight as neighbors."),
    Persona("Theo", "A 24-year-old barista and philosophy dropout. Contrarian, playful, loves poking holes in confident moral claims from any direction."),
    Persona("Sofia", "A 47-year-old army veteran turned logistics manager. Deeply loyal to her unit and family; believes you owe more to your own people than to strangers."),
]
