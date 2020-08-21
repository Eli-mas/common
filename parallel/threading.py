"""High-level routines for employing parallelism in code. At the moment
features only an interface around Python's threading functionality via the
Queuer class, but I can expand this with time."""

from threading import Thread
from queue import Queue, Empty
import traceback
from inspect import getfullargspec
from itertools import zip_longest
from ..common import print_update

def thread_queue_func(queue, func, result_list, global_kwargs, print_str=None):
	"""
	note: kwargs here is used when different keywords are passed
	to each call, not when the same keywords are passed to all calls
	in the latter case, the keywords are defined at the thread level
	within 'queuer'
	
	"""
	while True:
		try:
			i, (args, kwargs) = queue.get_nowait()
			#print('thread_queue_func:',i,args,kwargs,func)
			result_list[i] = func(*args, **global_kwargs, **kwargs)
			if print_str is not None:
				print_update(
					f'{sum(e is not None for e in result_list)} '
					f'/ {len(result_list)} {print_str}'
				)
			# this does not take into account errors that have left None being present in result_list
		except Empty:
			break
		except Exception as e:
			"""what should be done here -- break or pass, or something else?"""
			print('\nthread_queue_func exception',repr(e))
			traceback.print_exc()
			break
			pass#

def queuer(
	function, args_list, kwargs=None, iterkw=None, threads=5, print_str=None
):
	"""
	if a function is to be called repeatedly in a way that can be thread-parallelized,
	this function is a means to do so
	
	NOTE: iterkw functionality is not working; consider using IterDict
	
	would be nice to incorporate this functionality into a class
	
	this would allow for:
		making any number of these queue/threads pairs at a time
		enforcing a singleton pattern where desired
		having a Lock to synchronize an update mechanism
		...
	
	"""
	queue = Queue()
	result_list = [None]*len(args_list)
	if kwargs is None: kwargs = {}
	if print_str is None: print_str = 'completed'
	threads = [Thread(target=thread_queue_func,
					  args = (queue, function, result_list, kwargs, print_str))
					  for _ in range(threads)]
	
	if iterkw is None:
		for i,a in enumerate(args_list):
			queue.put((i,((a,),{})))
	else:
		for i,(a,k) in enumerate(zip(args_list,iterkw)):
			queue.put((i,((a,),k)))
	
	for t in threads: t.start()
	for t in threads: t.join()
	
	return result_list

class Queuer:
	def __init__(
		self, function, thread_count = 5, complexity = None,
		global_args=None, iterargs=None, iterkwargs=None, **global_kw):
		"""
		multithreading of a parallelizable function using a queue
		
		function:
			function to be multithreaded
		thread_count:
			how many threads to run; the actual number of threads will
			be smaller if fewer calls are to be made than the count specified
		complexity:
			if specified, an estimator of the time complexity of the inputs;
			this will be used to assign calls to groups so as to attempt
			to minimize overall run time
		"""
		self.function = function
		self.set_thread_count(thread_count)
		self.queue = Queue()
		self.global_kw = global_kw
		self.calls = []
		self.global_args = () if global_args is None else global_args
		self.add_calls(iterargs,iterkwargs)
	
	def set_thread_count(self,count):
		self.thread_count = count
	
	def add_calls(self,iterargs=None,iterkwargs=None):
		if (iterargs is None) and (iterkwargs is None): return
		if iterargs is None:
			self.calls.extend( zip_longest((), iterkwargs, fillvalue=() ))
		elif iterkwargs is None:
			self.calls.extend( zip_longest(iterargs, (), fillvalue={} ))
		else:
			self.calls.extend(zip(iterargs, iterkwargs))
	
	def add_call(self,*a,**kw):
		"""
		add a call to be made;
		the call will proceed as
		self.function(*a,**kw,**global_kw)
		"""
		self.calls.append((a,kw))
	
	def add_global_kw(self,**kw):
		self.global_kw.update(**kw)
	
	def pop_global_kw(self,**kw):
		return tuple(self.global_kw.pop(k) for k in kw)
	
	def clear_global_kw(self):
		"""
		clear global_kw but return a copy of it first
		"""
		k = {**self.global_kw}
		self.global_kw.clear()
		return k
	
	def _prepare(self):
		self.results = [None]*len(self.calls)
		self.threads = [
			Thread(
				target=thread_queue_func,
				args = (self.queue, self.function, self.results, self.global_kw))
				for _ in range(self.thread_count)
			]
		
		# the complexity estimator can be used
		# when I develop an algorithm for optimally partitioning
		# a set of inputs over n threads based on some scalar mapping of the inputs
		# note this will require restructuring how threads retrieve items from the queue
		# so that threads will be associated with specific call groups
		
		for i,args_kwargs in enumerate(self.calls):
			self.queue.put((i,args_kwargs))#(a,kw)
	
	def run(self):
		self._prepare()
		
		print('Queuer.run: calls:',self.calls)
		
		for t in self.threads: t.start()
		for t in self.threads: t.join()
		
		self.counter=0
		return self.results
		

__all__ = ('queuer','Queuer')

if __name__=='__main__':
	def f(l,m1=1,m2=1,a=0): return sum(l)*m1*m2+a
	
	q = Queuer(f,m2=2)
	q.add_call([1]*10,m1=1)
	q.add_call([10]*10,m1=2)
	q.add_call([100]*10,m1=3,a=1)
	assert q.run() == [20, 400, 6001]
