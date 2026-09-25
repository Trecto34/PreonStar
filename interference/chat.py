"""
Interactive Chat Interface for Q36 with Interference Search capabilities.
"""
import os
import readline
import sys
import time

from interference.backends.q36_backend import Q36Backend
from interference.backends.mock_backend import MockBackend
from interference.agent import run_agent_task
from interference.domains.countdown_domain import CountdownDomain
from interference.core import search


BANNER = r"""
  ___ _____   __   ___       _             __                              
 / _ \___ /  / /_ |_ _| _ _ | |_ ___ _ _ / _|___ _ _ ___ _ _  __ ___ ___ 
| (_) |_ \  | '_ \ | | | ' \|  _/ -_) '_|  _/ -_) '_/ -_) ' \/ _/ -_|_-< 
 \__\___/   |_.__/|___||_||_|\__\___|_| |_| \___|_| \___|_||_\__\___/__/ 
                  Interactive Agent with Multi-Branch Search
"""


class InterferenceChatSession:
    def __init__(self, backend_choice: str = "q36"):
        self.backend_choice = backend_choice
        if backend_choice == "q36":
            self.backend = Q36Backend()
        else:
            self.backend = MockBackend(lambda p: ["Olá! Sou o agente Q36 com busca por interferência."])
        self.history = []

    def handle_countdown(self, arg_str: str):
        # Format: 24 98 19 3 -> 361
        try:
            parts = arg_str.split("->")
            if len(parts) != 2:
                print("Uso: /countdown 24 98 19 3 -> 361")
                return
            nums = [int(x.strip()) for x in parts[0].split()]
            target = int(parts[1].strip())
            print(f"\n[Interference Search] Resolvendo Countdown: {nums} -> {target}...")
            domain = CountdownDomain(max_branch=12)
            t0 = time.perf_counter()
            res = search(domain, (nums, target), budget=200, width=6, merge=True)
            dt = (time.perf_counter() - t0) * 1000

            if res.solved and res.solution:
                for v, e in zip(res.solution.numbers, res.solution.exprs):
                    if v == target:
                        print(f"✅ Solução encontrada em {res.rounds} rodadas ({dt:.2f} ms): {e} = {target}")
                        print(f"   (Estados fundidos por interferência construtiva: {res.merged_away})")
            else:
                print("❌ Não foi possível encontrar uma solução exata.")
        except Exception as e:
            print(f"Erro ao processar: {e}")

    def handle_fix(self, arg_str: str):
        # Format: <file> ; <test_cmd> ; <task>
        parts = [p.strip() for p in arg_str.split(";")]
        if len(parts) < 3:
            print("Uso: /fix arquivo.py ; pytest tests/ ; Descrição da tarefa")
            return
        target_file, test_cmd, task = parts[0], parts[1], parts[2]
        if not os.path.exists(target_file):
            print(f"Erro: Arquivo '{target_file}' não encontrado.")
            return

        run_agent_task(
            target_file=target_file,
            test_cmd=test_cmd,
            task_instruction=task,
            backend_choice=self.backend_choice,
        )

    def chat_message(self, user_msg: str):
        print("\nQ36 pensando...", end="", flush=True)
        t0 = time.perf_counter()

        system_prompt = (
            "Você é o assistente de programação Q36 com capacidades de busca por interferência "
            "e raciocínio concorrente. Responda de forma direta, clara e técnica em português."
        )

        full_prompt = f"{system_prompt}\n\nUsuário: {user_msg}\nAssistente:"
        responses, tokens = self.backend.generate_candidates([full_prompt], max_tokens=1024)
        elapsed = time.perf_counter() - t0

        print(f"\r{' ' * 20}\r", end="")  # Clear thinking text
        if responses:
            reply = responses[0].strip()
            print(f"\n{reply}\n")
            speed = tokens / max(elapsed, 0.01)
            print(f"\033[90m[{tokens} tokens gerados em {elapsed:.2f}s (~{speed:.1f} t/s)]\033[0m\n")
        else:
            print("\n[Erro ao obter resposta do modelo]\n")

    def run(self):
        print(BANNER)
        print("Comandos disponíveis:")
        print("  /fix <arquivo> ; <comando_teste> ; <tarefa> -> Dispara o Interference Search no código")
        print("  /countdown <nums> -> <target>              -> Roda o desafio matemático de Countdown")
        print("  /clear                                      -> Limpa o terminal")
        print("  exit ou quit                                -> Sai do chat")
        print("-" * 75)
        print("Ou simplesmente converse normalmente digitando sua mensagem abaixo:\n")

        while True:
            try:
                user_input = input("\033[1;36mq36-interference>\033[0m ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", ":q"]:
                    print("\nEncerrando sessão. Até logo!")
                    break

                if user_input.startswith("/clear"):
                    os.system("clear")
                    continue

                if user_input.startswith("/countdown"):
                    self.handle_countdown(user_input[len("/countdown"):].strip())
                    continue

                if user_input.startswith("/fix"):
                    self.handle_fix(user_input[len("/fix"):].strip())
                    continue

                # Normal conversational prompt
                self.chat_message(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\nEncerrando sessão.")
                break


def main():
    backend = "q36"
    if "--mock" in sys.argv:
        backend = "mock"
    session = InterferenceChatSession(backend_choice=backend)
    session.run()


if __name__ == "__main__":
    main()
