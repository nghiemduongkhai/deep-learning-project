{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "vscode": {
     "languageId": "powershell"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#1. Write a Python program to calculate the length of a string.\n",
    "1+1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "vscode": {
     "languageId": "powershell"
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "A module that was compiled using NumPy 1.x cannot be run in\n",
      "NumPy 2.1.1 as it may crash. To support both 1.x and 2.x\n",
      "versions of NumPy, modules must be compiled with NumPy 2.0.\n",
      "Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.\n",
      "\n",
      "If you are a user of the module, the easiest solution will be to\n",
      "downgrade to 'numpy<2' or try to upgrade the affected module.\n",
      "We expect that some modules will need time to support NumPy 2.\n",
      "\n",
      "Traceback (most recent call last):  File \"<frozen runpy>\", line 198, in _run_module_as_main\n",
      "  File \"<frozen runpy>\", line 88, in _run_code\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel_launcher.py\", line 18, in <module>\n",
      "    app.launch_new_instance()\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/traitlets/config/application.py\", line 1075, in launch_instance\n",
      "    app.start()\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/kernelapp.py\", line 739, in start\n",
      "    self.io_loop.start()\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/tornado/platform/asyncio.py\", line 211, in start\n",
      "    self.asyncio_loop.run_forever()\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/asyncio/base_events.py\", line 641, in run_forever\n",
      "    self._run_once()\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/asyncio/base_events.py\", line 1986, in _run_once\n",
      "    handle._run()\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/asyncio/events.py\", line 88, in _run\n",
      "    self._context.run(self._callback, *self._args)\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/kernelbase.py\", line 545, in dispatch_queue\n",
      "    await self.process_one()\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/kernelbase.py\", line 534, in process_one\n",
      "    await dispatch(*args)\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/kernelbase.py\", line 437, in dispatch_shell\n",
      "    await result\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/ipkernel.py\", line 362, in execute_request\n",
      "    await super().execute_request(stream, ident, parent)\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/kernelbase.py\", line 778, in execute_request\n",
      "    reply_content = await reply_content\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/ipkernel.py\", line 449, in do_execute\n",
      "    res = shell.run_cell(\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/ipykernel/zmqshell.py\", line 549, in run_cell\n",
      "    return super().run_cell(*args, **kwargs)\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/IPython/core/interactiveshell.py\", line 3098, in run_cell\n",
      "    result = self._run_cell(\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/IPython/core/interactiveshell.py\", line 3153, in _run_cell\n",
      "    result = runner(coro)\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/IPython/core/async_helpers.py\", line 128, in _pseudo_sync_runner\n",
      "    coro.send(None)\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/IPython/core/interactiveshell.py\", line 3365, in run_cell_async\n",
      "    has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/IPython/core/interactiveshell.py\", line 3610, in run_ast_nodes\n",
      "    if await self.run_code(code, result, async_=asy):\n",
      "  File \"/Users/macbookpro/Library/Python/3.12/lib/python/site-packages/IPython/core/interactiveshell.py\", line 3670, in run_code\n",
      "    exec(code_obj, self.user_global_ns, self.user_ns)\n",
      "  File \"/var/folders/8f/j3jhljy518s53mrcxsrf2f240000gn/T/ipykernel_49917/4265195184.py\", line 1, in <module>\n",
      "    import torch\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/__init__.py\", line 1477, in <module>\n",
      "    from .functional import *  # noqa: F403\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/functional.py\", line 9, in <module>\n",
      "    import torch.nn.functional as F\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/nn/__init__.py\", line 1, in <module>\n",
      "    from .modules import *  # noqa: F403\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/nn/modules/__init__.py\", line 35, in <module>\n",
      "    from .transformer import TransformerEncoder, TransformerDecoder, \\\n",
      "  File \"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/nn/modules/transformer.py\", line 20, in <module>\n",
      "    device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),\n",
      "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)\n",
      "  device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),\n"
     ]
    }
   ],
   "source": [
    "import torch"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
