import Image from "next/image";

export default function Page()
{
    return (
        <>  
            <div className="flex flex-col justify-center gap-4 items-center  h-screen w-full bg-gradient-to-br from-gray-900 to-gray-500">
                <code className="text-lg font-semibold text-white">ML Model Output</code>
                <img src={'http://127.0.0.1:4500/output'}
                    alt="ML-Model-OUTPUT"
                    width={1090}
                    height={500}
                />
            </div>
        </>
    )
}