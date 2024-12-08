import os
import re
import json
import argparse
from tqdm import tqdm
import openai


class OpenAIModel:
    """
    A wrapper class for interaction with the OpenAI API.
    """

    def __init__(self, api_key: str = "your api key", model_version: str = "gpt-4o"):
        """
        Initialize the OpenAIModel instance.

        Args:
            api_key (str): Your OpenAI API key.
            model_version (str): The model version to be used (e.g., "gpt-4").
        """
        openai.api_key = api_key
        self.model = model_version

    def get_response(self, prompt: str, print_prompt: bool = False, temperature: float = 0.2, stream: bool = True) -> str:
        """
        Get a response from the OpenAI model.

        Args:
            prompt (str): The prompt text for the model.
            print_prompt (bool): If True, print the prompt before sending.
            temperature (float): The temperature parameter for sampling.
            stream (bool): Whether to stream the response as it arrives.

        Returns:
            str: The complete answer from the model.
        """
        if print_prompt:
            print(prompt)

        response_stream = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=stream,
        )

        answer = ""
        if stream:
            for chunk in response_stream:
                if 'choices' in chunk and chunk.choices and 'delta' in chunk.choices[0]:
                    content = chunk.choices[0].delta.get('content')
                    if content:
                        print(content, end="")
                        answer += content
            print()
        else:
            for chunk in response_stream:
                if 'choices' in chunk and chunk.choices and 'delta' in chunk.choices[0]:
                    content = chunk.choices[0].delta.get('content')
                    if content:
                        answer += content

        return answer


class Evaluator:
    """
    A class responsible for loading prompts, sending them to the OpenAI model, 
    and parsing the responses to extract evaluation scores.
    """

    def __init__(self, api_key: str = "your api key", model_version: str = "gpt-4o"):
        """
        Initialize the Evaluator instance with a configured OpenAI model.

        Args:
            api_key (str): Your OpenAI API key.
            model_version (str): The model version to be used.
        """
        self.model = OpenAIModel(api_key=api_key, model_version=model_version)

    def load_prompt(self, file_name: str, **kwargs) -> str:
        """
        Load a prompt template from a file and insert Q&A pairs.

        Args:
            file_name (str): The name of the file containing the base prompt.
            **kwargs: Arbitrary keyword arguments corresponding to Q&A from the data 
                      (e.g., q1, a1, q2, a2, ...).

        Returns:
            str: The constructed prompt string.
        """
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        loaded_text = ''.join(lines)
        
        qa_text = ""
        questions = {k: v for k, v in kwargs.items() if k.startswith('q')}
        answers = {k: v for k, v in kwargs.items() if k.startswith('a')}

        for q_key in sorted(questions.keys(), key=lambda x: int(x.replace('q', ''))):
            idx = q_key.replace('q', '')
            a_key = 'a' + idx
            question = questions[q_key]
            answer = answers.get(a_key, "")
            qa_text += f"질문 {idx}: {question}\n질문 {idx}의 답변: {answer}\n\n"

        prompt = loaded_text + qa_text
        prompt += "=" * 20 + "\n답변 점수:\n"
        return prompt

    def openai_eval(self, prompt: str) -> str:
        """
        Send a prompt to the OpenAI model and get the response.

        Args:
            prompt (str): The constructed prompt string.

        Returns:
            str: The model's response.
        """
        response = self.model.get_response(prompt)
        return response

    def parse_response(self, response: str) -> dict:
        """
        Parse the model response to extract scores and reasoning for each question.

        The expected response format:
        질문 N의 답변 점수:
        카테고리점수: X
        이유: Y
        ... (repeated for each category)

        Args:
            response (str): The model response text.

        Returns:
            dict: A dictionary mapping each question to its categories, 
                  scores, and reasons.
        """
        data = {}
        question_pattern = r"(질문\s*\d+\s*의\s*답변\s*점수\s*:)(.*?)(?=질문\s*\d+\s*의\s*답변\s*점수\s*:|$)"
        question_matches = re.finditer(question_pattern, response, re.DOTALL)

        for question_match in question_matches:
            question_header = question_match.group(1).strip()
            question_content = question_match.group(2).strip()
            question_number = re.search(r"\d+", question_header).group()

            scores = {}
            pattern = (
                r"(?P<category>^[^\n]+점수)\s*:\s*(?P<score>\d+)\s*"
                r"(?:\n|\s)*이유\s*:\s*(?P<reason>.*?)(?=\n[^\n]+점수\s*:\s*\d+|$)"
            )
            matches = re.finditer(pattern, question_content, re.DOTALL | re.MULTILINE)
            for match in matches:
                category = match.group('category').strip()
                score = int(match.group('score'))
                reason = match.group('reason').strip()
                scores[category] = {
                    '점수': score,
                    '이유': reason
                }

            data[f"질문 {question_number}"] = scores

        return data


def run(cfg):
    """
    Run the evaluation process:
    1. Load data from a JSON file.
    2. Use the Evaluator to generate prompts and get model responses.
    3. Parse responses and store the scores in a JSON file.

    Args:
        cfg (argparse.Namespace): An object with attributes including 
                                  'category' indicating the data category.
    """
    evaluator = Evaluator()

    with open(os.path.join('data', 'final_self_introduction_classified.json'), 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)

    score_dict_path = os.path.join('data', f'{cfg.category}_score_dict.json')
    if os.path.exists(score_dict_path):
        with open(score_dict_path, 'r', encoding='utf-8') as f:
            score_dict = json.load(f)
    else:
        score_dict = {}

    for key, item in tqdm(sampled_data[cfg.category].items(), desc="Evaluating..."):
        if key in score_dict:
            continue

        params = {k: v for k, v in item.items() if k.startswith('q') or k.startswith('a')}

        prompt = evaluator.load_prompt('prompt_final.txt', **params)
        response = evaluator.openai_eval(prompt)
        scores = evaluator.parse_response(response)
        score_dict[key] = scores

        with open(score_dict_path, 'w', encoding='utf-8') as f:
            json.dump(score_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Q&A responses using OpenAI model.")
    parser.add_argument("--category", type=str, default="category", help="Category key to evaluate.")
    cfg = parser.parse_args()
    run(cfg)