document.addEventListener('DOMContentLoaded', () => {
    
    // Consts:
    
    const valor10  = document.querySelector("#valor10");
    const valor50  = document.querySelector("#valor50");
    const valor100 = document.querySelector("#valor100");

    const valorT   = document.querySelector("#total");

    // Functions:
    
    async function atualizarValores() {
        try {
            const response = await fetch('/data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            // Atualiza os valores na tela
            // Note que as chaves do JSON (vindas do Python) são strings
            valor10.textContent  = data.counts["10"]  || 0;
            valor50.textContent  = data.counts["50"]  || 0;
            valor100.textContent = data.counts["100"] || 0;

            // Formata o total como moeda BRL (R$)
            valorT.textContent = data.total.toLocaleString('pt-BR', {
                style: 'currency',
                currency: 'BRL'
            });

        } catch (error) {
            console.error("Erro ao buscar dados de contagem:", error);
        }
    }

    setInterval(atualizarValores, 500);
})